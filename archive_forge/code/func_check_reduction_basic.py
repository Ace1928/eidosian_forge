from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def check_reduction_basic(self, pyfunc, **kwargs):
    cfunc = jit(nopython=True)(pyfunc)

    def check(arr):
        self.assertPreciseEqual(pyfunc(arr), cfunc(arr), **kwargs)
    arr = np.float64([1.0, 2.0, 0.0, -0.0, 1.0, -1.5])
    check(arr)
    arr = np.float64([-0.0, -1.5])
    check(arr)
    arr = np.float64([-1.5, 2.5, 'inf'])
    check(arr)
    arr = np.float64([-1.5, 2.5, '-inf'])
    check(arr)
    arr = np.float64([-1.5, 2.5, 'inf', '-inf'])
    check(arr)
    arr = np.float64(['nan', -1.5, 2.5, 'nan', 3.0])
    check(arr)
    arr = np.float64(['nan', -1.5, 2.5, 'nan', 'inf', '-inf', 3.0])
    check(arr)
    arr = np.float64([5.0, 'nan', -1.5, 'nan'])
    check(arr)
    arr = np.float64(['nan', 'nan'])
    check(arr)