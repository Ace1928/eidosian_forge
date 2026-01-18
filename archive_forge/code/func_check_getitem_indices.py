import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase
def check_getitem_indices(self, arr_shape, index):

    @njit
    def numba_get_item(array, idx):
        return array[idx]
    arr = np.random.randint(0, 11, size=arr_shape)
    get_item = numba_get_item.py_func
    orig_base = arr.base or arr
    expected = get_item(arr, index)
    got = numba_get_item(arr, index)
    self.assertIsNot(expected.base, orig_base)
    self.assertEqual(got.shape, expected.shape)
    self.assertEqual(got.dtype, expected.dtype)
    np.testing.assert_equal(got, expected)
    self.assertFalse(np.may_share_memory(got, expected))