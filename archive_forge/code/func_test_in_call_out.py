import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_in_call_out(self):
    """Function is passed in as an argument, called, and returned.
        """

    def a(i):
        return i + 1

    def foo(f):
        f(123)
        return f
    sig = int64(int64)
    for decor in [mk_cfunc_func(sig), njit_func, mk_njit_with_sig_func(sig), mk_wap_func(sig)]:
        for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
            jit_ = jit(**jit_opts)
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                r1 = jit_(foo)(a_).pyfunc
                r2 = foo(a)
                self.assertEqual(r1, r2)