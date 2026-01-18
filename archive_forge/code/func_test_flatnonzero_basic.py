from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_flatnonzero_basic(self):
    pyfunc = numpy_flatnonzero
    cfunc = jit(nopython=True)(pyfunc)

    def a_variations():
        yield np.arange(-5, 5)
        yield np.full(5, fill_value=0)
        yield np.array([])
        a = self.random.randn(100)
        a[np.abs(a) > 0.2] = 0.0
        yield a
        yield a.reshape(5, 5, 4)
        yield a.reshape(50, 2, order='F')
        yield a.reshape(25, 4)[1::2]
        yield (a * 1j)
    for a in a_variations():
        expected = pyfunc(a)
        got = cfunc(a)
        self.assertPreciseEqual(expected, got)