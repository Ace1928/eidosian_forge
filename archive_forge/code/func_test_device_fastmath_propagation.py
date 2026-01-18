from typing import List
from dataclasses import dataclass, field
from numba import cuda, float32
from numba.cuda.compiler import compile_ptx_for_current_device, compile_ptx
from math import cos, sin, tan, exp, log, log10, log2, pow, tanh
from operator import truediv
import numpy as np
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
import unittest
@unittest.expectedFailure
def test_device_fastmath_propagation(self):

    @cuda.jit('float32(float32, float32)', device=True)
    def foo(a, b):
        return a / b

    def bar(arr, val):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = foo(i, val)
    sig = (float32[::1], float32)
    fastver = cuda.jit(sig, fastmath=True)(bar)
    precver = cuda.jit(sig)(bar)
    self.assertIn('div.approx.f32', fastver.inspect_asm(sig))
    self.assertIn('div.rn.f32', precver.inspect_asm(sig))
    self.assertNotIn('div.approx.f32', precver.inspect_asm(sig))
    self.assertNotIn('div.full.f32', precver.inspect_asm(sig))