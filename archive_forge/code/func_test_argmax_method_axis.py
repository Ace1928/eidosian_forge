from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_argmax_method_axis(self):
    arr2d = np.arange(6).reshape(2, 3)

    def argmax(arr):
        return arr2d.argmax(axis=0)
    self.assertPreciseEqual(argmax(arr2d), jit(nopython=True)(argmax)(arr2d))