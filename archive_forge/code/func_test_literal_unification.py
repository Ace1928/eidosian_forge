import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_literal_unification(self):

    @jit(nopython=True)
    def Data1(value):
        return Rect(value, -321)

    @jit(nopython=True)
    def call(i, j):
        if j == 0:
            result = Data1(i)
        else:
            result = Rect(i, j)
        return result
    r = call(123, 1321)
    self.assertEqual(r, Rect(width=123, height=1321))
    r = call(123, 0)
    self.assertEqual(r, Rect(width=123, height=-321))