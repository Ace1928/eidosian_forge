import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_in_ns_seq_call(self):
    """Functions are passed in as an argument and via namespace scoping
        (mixed pathways), used as tuple items, and called.

        """

    def a(i):
        return i + 1

    def b(i):
        return i + 2

    def mkfoo(b_):

        def foo(f):
            r = 0
            for f_ in (f, b_):
                r = r + f_(r)
            return r
        return foo
    sig = int64(int64)
    for decor in [mk_cfunc_func(sig), mk_njit_with_sig_func(sig), mk_wap_func(sig), mk_ctypes_func(sig)][:-1]:
        for jit_opts in [dict(nopython=True), dict(forceobj=True)]:
            jit_ = jit(**jit_opts)
            with self.subTest(decor=decor.__name__):
                a_ = decor(a)
                b_ = decor(b)
                self.assertEqual(jit_(mkfoo(b_))(a_), mkfoo(b)(a))