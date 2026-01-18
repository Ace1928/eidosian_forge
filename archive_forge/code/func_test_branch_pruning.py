import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_branch_pruning(self):

    @njit
    def foo(rec, flag=None):
        n = 0
        n += rec['a']
        if flag is not None:
            n += rec['b']
            rec['b'] += 20
        return n
    self.assertEqual(foo(self.a_rec1), self.a_rec1[0])
    k = self.ab_rec1[1]
    self.assertEqual(foo(self.ab_rec1, flag=1), self.ab_rec1[0] + k)
    self.assertEqual(self.ab_rec1[1], k + 20)
    foo.disable_compile()
    self.assertEqual(len(foo.nopython_signatures), 2)
    self.assertEqual(foo(self.a_rec1) + 1, foo(self.ab_rec1))
    self.assertEqual(foo(self.ab_rec1, flag=1), self.ab_rec1[0] + k + 20)