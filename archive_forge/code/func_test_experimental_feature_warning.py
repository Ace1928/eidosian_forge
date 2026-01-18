import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_experimental_feature_warning(self):

    @jit(nopython=True)
    def more(x):
        return x + 1

    @jit(nopython=True)
    def less(x):
        return x - 1

    @jit(nopython=True)
    def foo(sel, x):
        fn = more if sel else less
        return fn(x)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        res = foo(True, 10)
    self.assertEqual(res, 11)
    self.assertEqual(foo(False, 10), 9)
    self.assertGreaterEqual(len(ws), 1)
    pat = 'First-class function type feature is experimental'
    for w in ws:
        if pat in str(w.message):
            break
    else:
        self.fail('missing warning')