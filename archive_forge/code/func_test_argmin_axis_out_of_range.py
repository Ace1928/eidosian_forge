from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_argmin_axis_out_of_range(self):
    arr1d = np.arange(6)
    arr2d = np.arange(6).reshape(2, 3)

    @jit(nopython=True)
    def jitargmin(arr, axis):
        return np.argmin(arr, axis)

    def assert_raises(arr, axis):
        with self.assertRaisesRegex(ValueError, 'axis.*out of bounds'):
            jitargmin.py_func(arr, axis)
        with self.assertRaisesRegex(ValueError, 'axis.*out of bounds'):
            jitargmin(arr, axis)
    assert_raises(arr1d, 1)
    assert_raises(arr1d, -2)
    assert_raises(arr2d, -3)
    assert_raises(arr2d, 2)
    self.disable_leak_check()