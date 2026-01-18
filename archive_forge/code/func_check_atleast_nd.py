from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def check_atleast_nd(self, pyfunc, cfunc):

    def check_result(got, expected):
        self.assertStridesEqual(got, expected)
        self.assertPreciseEqual(got.flatten(), expected.flatten())

    def check_single(arg):
        check_result(cfunc(arg), pyfunc(arg))

    def check_tuple(*args):
        expected_tuple = pyfunc(*args)
        got_tuple = cfunc(*args)
        self.assertEqual(len(got_tuple), len(expected_tuple))
        for got, expected in zip(got_tuple, expected_tuple):
            check_result(got, expected)
    a1 = np.array(42)
    a2 = np.array(5j)
    check_single(a1)
    check_tuple(a1, a2)
    b1 = np.arange(5)
    b2 = np.arange(6) + 1j
    b3 = b1[::-1]
    check_single(b1)
    check_tuple(b1, b2, b3)
    c1 = np.arange(6).reshape((2, 3))
    c2 = c1.T
    c3 = c1[::-1]
    check_single(c1)
    check_tuple(c1, c2, c3)
    d1 = np.arange(24).reshape((2, 3, 4))
    d2 = d1.T
    d3 = d1[::-1]
    check_single(d1)
    check_tuple(d1, d2, d3)
    e = np.arange(16).reshape((2, 2, 2, 2))
    check_single(e)
    check_tuple(a1, b2, c3, d2)