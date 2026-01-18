import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc
def test_dufunc_invalid_axis(self):
    duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)

    @njit
    def foo(a, axis):
        return duadd.reduce(a, axis=axis)
    a = np.arange(40).reshape(5, 4, 2)
    cases = ((0, 0), (0, 1, 0), (0, -3), (-1, -1), (-1, 2))
    for axis in cases:
        msg = "duplicate value in 'axis'"
        with self.assertRaisesRegex(ValueError, msg):
            foo(a, axis)
    cases = (-4, 3, (0, -4))
    for axis in cases:
        with self.assertRaisesRegex(ValueError, 'Invalid axis'):
            foo(a, axis)