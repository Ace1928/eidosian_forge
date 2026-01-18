import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_array_view_iter(self):
    arr = np.arange(12).reshape((3, 4))
    self.check_array_view_iter(arr, 1)
    self.check_array_view_iter(arr.T, 1)
    arr = arr[::2]
    self.check_array_view_iter(arr, 1)
    arr = np.bool_([1, 0, 0, 1]).reshape((2, 2))
    self.check_array_view_iter(arr, 1)