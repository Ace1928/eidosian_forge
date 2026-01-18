import numpy as np
from numba.cuda import compile_ptx
from numba.core.types import f2, i1, i2, i4, i8, u1, u2, u4, u8
from numba import cuda
from numba.core import types
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
from numba.types import float16, float32
import itertools
import unittest
@skip_on_cudasim('Compilation unsupported in the simulator')
def test_native_cast(self):
    float32_ptx, _ = cuda.compile_ptx(native_cast, (float32,), device=True)
    self.assertIn('st.f32', float32_ptx)
    float16_ptx, _ = cuda.compile_ptx(native_cast, (float16,), device=True)
    self.assertIn('st.u16', float16_ptx)