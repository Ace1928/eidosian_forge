import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase
def check_setitem_indices(self, arr_shape, index):

    @njit
    def set_item(array, idx, item):
        array[idx] = item
    arr = np.random.randint(0, 11, size=arr_shape)
    src = arr[index]
    expected = np.zeros_like(arr)
    got = np.zeros_like(arr)
    set_item.py_func(expected, index, src)
    set_item(got, index, src)
    self.assertEqual(got.shape, expected.shape)
    self.assertEqual(got.dtype, expected.dtype)
    np.testing.assert_equal(got, expected)