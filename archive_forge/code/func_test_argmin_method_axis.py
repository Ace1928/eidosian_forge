from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_argmin_method_axis(self):
    arr2d = np.arange(6).reshape(2, 3)

    def argmin(arr):
        return arr2d.argmin(axis=0)
    self.assertPreciseEqual(argmin(arr2d), jit(nopython=True)(argmin)(arr2d))