from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_expand_dims(self):
    pyfunc = expand_dims
    cfunc = njit(pyfunc)

    def check(arr, axis):
        expected = pyfunc(arr, axis)
        self.memory_leak_setup()
        got = cfunc(arr, axis)
        self.assertPreciseEqual(got, expected)
        del got
        self.memory_leak_teardown()

    def check_all_axes(arr):
        for axis in range(-arr.ndim - 1, arr.ndim + 1):
            check(arr, axis)
    arr = np.arange(5)
    check_all_axes(arr)
    arr = np.arange(24).reshape((2, 3, 4))
    check_all_axes(arr)
    check_all_axes(arr.T)
    check_all_axes(arr[::-1])
    arr = np.array(42)
    check_all_axes(arr)