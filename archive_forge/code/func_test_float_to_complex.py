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
def test_float_to_complex(self):
    pyfuncs = (to_complex64, to_complex128)
    totys = (np.complex64, np.complex128)
    fromtys = (np.float16, np.float32, np.float64)
    for pyfunc, toty in zip(pyfuncs, totys):
        for fromty in fromtys:
            with self.subTest(fromty=fromty, toty=toty):
                cfunc = self._create_wrapped(pyfunc, fromty, toty)
                np.testing.assert_allclose(cfunc(3.21), pyfunc(fromty(3.21)))
                np.testing.assert_allclose(cfunc(-3.21), pyfunc(fromty(-3.21)) + 0j)