// RUN: %clangxx -fsycl-device-only %s -o %t.bc
//
//==----------- spec_const_accessors.cpp -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The test checks that Clang doesn't crash if multiple accessors are used
// together with specialization constants in single kernel.

#include <CL/sycl.hpp>

using namespace sycl;

class MyInt32Const;
class MyKernel;

int32_t val = 10;

int main() {
  cl::sycl::queue queue;
  cl::sycl::program program(queue.get_context());

  cl::sycl::experimental::spec_constant<int32_t, MyInt32Const> i32 =
      program.set_spec_constant<MyInt32Const>(val);

  program.build_with_kernel_type<MyKernel>();
  int32_t a = 0;
  int32_t b = 0;
  cl::sycl::buffer<int32_t, 1> buf_a(&a, 1);
  cl::sycl::buffer<int32_t, 1> buf_b(&b, 1);

  queue.submit([&](cl::sycl::handler &cgh) {
    auto acc_a = buf_a.get_access<cl::sycl::access::mode::write>(cgh);
    auto acc_b = buf_b.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.single_task<MyKernel>(
        program.get_kernel<MyKernel>(),
        [=]() {
          acc_a[0] = i32.get();
          acc_b[0] = i32.get();
        });
  });
  return 0;
}
