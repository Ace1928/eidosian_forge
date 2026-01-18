import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_in_overload(self):
    """Function is passed in as an argument and called with different
        argument types.

        """

    def a(i):
        return i + 1

    def foo(f):
        r1 = f(123)
        r2 = f(123.45)
        return (r1, r2)
    for decor in [njit_func]:
        for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
            jit_ = jit(**jit_opts)
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                self.assertEqual(jit_(foo)(a_), foo(a))