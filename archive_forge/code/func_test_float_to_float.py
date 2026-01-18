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
def test_float_to_float(self):
    pyfuncs = (to_float16, to_float32, to_float64)
    tys = (np.float16, np.float32, np.float64)
    for (pyfunc, fromty), toty in itertools.product(zip(pyfuncs, tys), tys):
        with self.subTest(fromty=fromty, toty=toty):
            cfunc = self._create_wrapped(pyfunc, fromty, toty)
            np.testing.assert_allclose(cfunc(12.3), toty(12.3) / toty(2), rtol=0.0003)
            np.testing.assert_allclose(cfunc(-12.3), toty(-12.3) / toty(2), rtol=0.0003)