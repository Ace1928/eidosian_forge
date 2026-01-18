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
def test_itemset(self):
    pyfunc = array_itemset
    cfunc = jit(nopython=True)(pyfunc)

    def check_ok(a, v):
        expected = a.copy()
        got = a.copy()
        pyfunc(expected, v)
        cfunc(got, v)
        self.assertPreciseEqual(got, expected)

    def check_err(a):
        with self.assertRaises(ValueError) as raises:
            cfunc(a, 42)
        self.assertIn('itemset(): can only write to an array of size 1', str(raises.exception))
    self.disable_leak_check()
    check_ok(np.float32([1.5]), 42)
    check_ok(np.complex128([[1.5j]]), 42)
    check_ok(np.array(1.5), 42)
    check_err(np.array([1, 2]))
    check_err(np.array([]))