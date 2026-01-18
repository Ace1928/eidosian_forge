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
def test_arange_3_arg(self):
    windows64 = sys.platform.startswith('win32') and sys.maxsize > 2 ** 32

    def check_ok(arg0, arg1, arg2, pyfunc, cfunc, check_dtype=False):
        expected = pyfunc(arg0, arg1, arg2)
        got = cfunc(arg0, arg1, arg2)
        np.testing.assert_allclose(expected, got)
        if not windows64:
            self.assertEqual(expected.dtype, got.dtype)
    for pyfunc in (np_arange_3, np_arange_2_step, np_arange_start_stop_step):
        cfunc = jit(nopython=True)(pyfunc)
        check_ok(0, 5, 1, pyfunc, cfunc)
        check_ok(-8, -1, 3, pyfunc, cfunc)
        check_ok(0, -10, -2, pyfunc, cfunc)
        check_ok(0.5, 4, 2, pyfunc, cfunc)
        check_ok(0, 1, 0.1, pyfunc, cfunc)
        check_ok(0, complex(4, 4), complex(1, 1), pyfunc, cfunc)
        check_ok(3, 6, None, pyfunc, cfunc)
        check_ok(3, None, None, pyfunc, cfunc)
        check_ok(np.int8(0), np.int8(5), np.int8(1), pyfunc, cfunc)
        check_ok(np.int8(0), np.int16(5), np.int32(1), pyfunc, cfunc)
        i8 = np.int8
        check_ok(i8(0), i8(5), i8(1), pyfunc, cfunc, True)
        check_ok(np.int64(0), i8(5), i8(1), pyfunc, cfunc, True)
    pyfunc = np_arange_2_dtype
    cfunc = jit(nopython=True)(pyfunc)
    check_ok(1, 5, np.float32, pyfunc, cfunc)
    check_ok(2.0, 8, np.int32, pyfunc, cfunc)
    check_ok(-2, 10, np.complex128, pyfunc, cfunc)
    check_ok(3, np.complex64(10), np.complex128, pyfunc, cfunc)
    check_ok(1, 7, None, pyfunc, cfunc)
    check_ok(np.int8(0), np.int32(5), None, pyfunc, cfunc, True)