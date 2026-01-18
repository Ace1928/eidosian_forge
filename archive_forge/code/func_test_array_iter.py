import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_array_iter(self):
    arr = np.arange(6)
    self.check_array_iter_1d(arr)
    self.check_array_iter_items(arr)
    arr = arr[::2]
    self.assertFalse(arr.flags.c_contiguous)
    self.assertFalse(arr.flags.f_contiguous)
    self.check_array_iter_1d(arr)
    self.check_array_iter_items(arr)
    arr = np.bool_([1, 0, 0, 1])
    self.check_array_iter_1d(arr)
    self.check_array_iter_items(arr)
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    self.check_array_iter_items(arr)
    self.check_array_iter_items(arr.T)