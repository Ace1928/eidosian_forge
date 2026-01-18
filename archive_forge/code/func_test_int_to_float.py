import numpy as np
from numba.cuda import compile_ptx
from numba.core.types import f2, i1, i2, i4, i8, u1, u2, u4, u8
from numba import cuda
from numba.core import types
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
from numba.types import float16, float32
import itertools
import unittest
@skip_unless_cc_53
def test_int_to_float(self):
    pyfuncs = (to_float16, to_float32, to_float64)
    totys = (np.float16, np.float32, np.float64)
    for pyfunc, toty in zip(pyfuncs, totys):
        with self.subTest(toty=toty):
            cfunc = self._create_wrapped(pyfunc, np.int64, toty)
            self.assertEqual(cfunc(321), pyfunc(321))