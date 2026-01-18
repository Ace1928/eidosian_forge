from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
def test_arange_1_arg(self):
    all_pyfuncs = (np_arange_1, lambda x: np.arange(x, 10), lambda x: np.arange(7, step=max(1, abs(x))))
    for pyfunc in all_pyfuncs:
        cfunc = jit(nopython=True)(pyfunc)

        def check_ok(arg0):
            expected = pyfunc(arg0)
            got = cfunc(arg0)
            np.testing.assert_allclose(expected, got)
        check_ok(0)
        check_ok(1)
        check_ok(4)
        check_ok(5.5)
        check_ok(-3)
        check_ok(complex(4, 4))
        check_ok(np.int8(0))