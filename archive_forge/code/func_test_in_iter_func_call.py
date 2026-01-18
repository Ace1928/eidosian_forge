import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_in_iter_func_call(self):
    """Functions are passed in as items of tuple argument, retrieved via
        indexing, and called within a variable for-loop.

        """

    def a(i):
        return i + 1

    def b(i):
        return i + 2

    def foo(funcs, n):
        r = 0
        for i in range(n):
            f = funcs[i]
            r = r + f(r)
        return r
    sig = int64(int64)
    for decor in [mk_cfunc_func(sig), mk_wap_func(sig), mk_njit_with_sig_func(sig)]:
        for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
            jit_ = jit(**jit_opts)
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                b_ = decor(b)
                self.assertEqual(jit_(foo)((a_, b_), 2), foo((a, b), 2))