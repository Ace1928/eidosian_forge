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
def test_item(self):
    pyfunc = array_item
    cfunc = jit(nopython=True)(pyfunc)

    def check_ok(arg):
        expected = pyfunc(arg)
        got = cfunc(arg)
        self.assertPreciseEqual(got, expected)

    def check_err(arg):
        with self.assertRaises(ValueError) as raises:
            cfunc(arg)
        self.assertIn('item(): can only convert an array of size 1 to a Python scalar', str(raises.exception))
    self.disable_leak_check()
    check_ok(np.float32([1.5]))
    check_ok(np.complex128([[1.5j]]))
    check_ok(np.array(1.5))
    check_ok(np.bool_(True))
    check_ok(np.float32(1.5))
    check_err(np.array([1, 2]))
    check_err(np.array([]))