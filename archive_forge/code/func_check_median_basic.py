from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def check_median_basic(self, pyfunc, array_variations):
    cfunc = jit(nopython=True)(pyfunc)

    def check(arr):
        expected = pyfunc(arr)
        got = cfunc(arr)
        self.assertPreciseEqual(got, expected)

    def check_odd(a):
        check(a)
        a = a.reshape((9, 7))
        check(a)
        check(a.T)
    for a in array_variations(np.arange(63) + 10.5):
        check_odd(a)

    def check_even(a):
        check(a)
        a = a.reshape((4, 16))
        check(a)
        check(a.T)
    for a in array_variations(np.arange(64) + 10.5):
        check_even(a)