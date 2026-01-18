import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_array_flat_len(self):
    pyfunc = array_flat_len
    cfunc = njit(array_flat_len)

    def check(arr):
        expected = pyfunc(arr)
        self.assertPreciseEqual(cfunc(arr), expected)
    arr = np.arange(24).reshape(4, 2, 3)
    check(arr)
    arr = arr.T
    check(arr)
    arr = arr[::2]
    check(arr)
    arr = np.array([42]).reshape(())
    check(arr)