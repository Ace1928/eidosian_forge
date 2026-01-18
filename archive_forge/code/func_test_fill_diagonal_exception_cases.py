from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_fill_diagonal_exception_cases(self):
    pyfunc = numpy_fill_diagonal
    cfunc = jit(nopython=True)(pyfunc)
    val = 1
    self.disable_leak_check()
    for a in (np.array([]), np.ones(5)):
        with self.assertRaises(TypingError) as raises:
            cfunc(a, val)
        assert 'The first argument must be at least 2-D' in str(raises.exception)
    with self.assertRaises(ValueError) as raises:
        a = np.zeros((3, 3, 4))
        cfunc(a, val)
        self.assertEqual('All dimensions of input must be of equal length', str(raises.exception))

    def _assert_raises(arr, val):
        with self.assertRaises(ValueError) as raises:
            cfunc(arr, val)
        self.assertEqual('Unable to safely conform val to a.dtype', str(raises.exception))
    arr = np.zeros((3, 3), dtype=np.int32)
    val = np.nan
    _assert_raises(arr, val)
    val = [3.3, np.inf]
    _assert_raises(arr, val)
    val = np.array([1, 2, 10000000000.0], dtype=np.int64)
    _assert_raises(arr, val)
    arr = np.zeros((3, 3), dtype=np.float32)
    val = [1.4, 2.6, -1e+100]
    _assert_raises(arr, val)
    val = 1.1e+100
    _assert_raises(arr, val)
    val = np.array([-1e+100])
    _assert_raises(arr, val)