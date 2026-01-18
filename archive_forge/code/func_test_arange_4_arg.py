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
def test_arange_4_arg(self):
    for pyfunc in (np_arange_4, np_arange_start_stop_step_dtype):
        cfunc = jit(nopython=True)(pyfunc)

        def check_ok(arg0, arg1, arg2, arg3):
            expected = pyfunc(arg0, arg1, arg2, arg3)
            got = cfunc(arg0, arg1, arg2, arg3)
            np.testing.assert_allclose(expected, got)
        check_ok(0, 5, 1, np.float64)
        check_ok(-8, -1, 3, np.int32)
        check_ok(0, -10, -2, np.float32)
        check_ok(0.5, 4, 2, None)
        check_ok(0, 1, 0.1, np.complex128)
        check_ok(0, complex(4, 4), complex(1, 1), np.complex128)
        check_ok(3, 6, None, None)
        check_ok(3, None, None, None)