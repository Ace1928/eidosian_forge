from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
def test_literal_value_passthrough(self):

    def bar(x):
        pass

    @overload(bar)
    def ol_bar(x):
        self.assertTrue(isinstance(x, types.LiteralList))
        lv = x.literal_value
        self.assertTrue(isinstance(lv, list))
        self.assertEqual(lv[0], types.literal(1))
        self.assertEqual(lv[1], types.literal('a'))
        self.assertEqual(lv[2], types.Array(types.float64, 1, 'C'))
        self.assertEqual(lv[3], types.List(types.intp, reflected=False, initial_value=[1, 2, 3]))
        self.assertTrue(isinstance(lv[4], types.LiteralList))
        self.assertEqual(lv[4].literal_value[0], types.literal('cat'))
        self.assertEqual(lv[4].literal_value[1], types.literal(10))
        return lambda x: x

    @njit
    def foo():
        otherhomogeneouslist = [1, 2, 3]
        otherheterogeneouslist = ['cat', 10]
        zeros = np.zeros(5)
        l = [1, 'a', zeros, otherhomogeneouslist, otherheterogeneouslist]
        bar(l)
    foo()