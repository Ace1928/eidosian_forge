import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_array_flat_3d(self):
    arr = np.arange(24).reshape(4, 2, 3)
    arrty = typeof(arr)
    self.assertEqual(arrty.ndim, 3)
    self.assertEqual(arrty.layout, 'C')
    self.assertTrue(arr.flags.c_contiguous)
    self.check_array_flat(arr)
    arr = arr.transpose()
    self.assertFalse(arr.flags.c_contiguous)
    self.assertTrue(arr.flags.f_contiguous)
    self.assertEqual(typeof(arr).layout, 'F')
    self.check_array_flat(arr)
    arr = arr[::2]
    self.assertFalse(arr.flags.c_contiguous)
    self.assertFalse(arr.flags.f_contiguous)
    self.assertEqual(typeof(arr).layout, 'A')
    self.check_array_flat(arr)
    arr = np.bool_([1, 0, 0, 1] * 2).reshape((2, 2, 2))
    self.check_array_flat(arr)