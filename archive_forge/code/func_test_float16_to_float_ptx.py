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
def test_float16_to_float_ptx(self):
    pyfuncs = (to_float32, to_float64)
    postfixes = ('f32', 'f64')
    for pyfunc, postfix in zip(pyfuncs, postfixes):
        ptx, _ = compile_ptx(pyfunc, (f2,), device=True)
        self.assertIn(f'cvt.{postfix}.f16', ptx)