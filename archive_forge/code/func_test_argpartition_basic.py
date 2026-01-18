import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_argpartition_basic(self):
    pyfunc = argpartition
    cfunc = jit(nopython=True)(pyfunc)
    d = np.array([], dtype=np.int64)
    expected = pyfunc(d, 0)
    got = cfunc(d, 0)
    self.assertPreciseEqual(expected, got)
    d = np.ones(1, dtype=np.int64)
    expected = pyfunc(d, 0)
    got = cfunc(d, 0)
    self.assertPreciseEqual(expected, got)
    kth = np.array([30, 15, 5])
    okth = kth.copy()
    cfunc(np.arange(40), kth)
    self.assertPreciseEqual(kth, okth)
    for r in ([2, 1], [1, 2], [1, 1]):
        d = np.array(r)
        tgt = np.argsort(d)
        for k in (0, 1):
            self.assertPreciseEqual(d[cfunc(d, k)[k]], d[tgt[k]])
            self.argpartition_sanity_check(pyfunc, cfunc, d, k)
    for r in ([3, 2, 1], [1, 2, 3], [2, 1, 3], [2, 3, 1], [1, 1, 1], [1, 2, 2], [2, 2, 1], [1, 2, 1]):
        d = np.array(r)
        tgt = np.argsort(d)
        for k in (0, 1, 2):
            self.assertPreciseEqual(d[cfunc(d, k)[k]], d[tgt[k]])
            self.argpartition_sanity_check(pyfunc, cfunc, d, k)
    d = np.ones(50)
    self.assertPreciseEqual(d[cfunc(d, 0)], d)
    d = np.arange(49)
    for k in (5, 15):
        self.assertEqual(cfunc(d, k)[k], k)
        self.partition_sanity_check(pyfunc, cfunc, d, k)
    d = np.arange(47)[::-1]
    for a in (d, d.tolist(), tuple(d.tolist())):
        self.assertEqual(cfunc(a, 6)[6], 40)
        self.assertEqual(cfunc(a, 16)[16], 30)
        self.assertPreciseEqual(cfunc(a, -6), cfunc(a, 41))
        self.assertPreciseEqual(cfunc(a, -16), cfunc(a, 31))
        self.argpartition_sanity_check(pyfunc, cfunc, d, -16)
    d = np.arange(1000000)
    x = np.roll(d, d.size // 2)
    mid = x.size // 2 + 1
    self.assertEqual(x[cfunc(x, mid)[mid]], mid)
    d = np.arange(1000001)
    x = np.roll(d, d.size // 2 + 1)
    mid = x.size // 2 + 1
    self.assertEqual(x[cfunc(x, mid)[mid]], mid)
    d = np.ones(10)
    d[1] = 4
    self.assertEqual(d[cfunc(d, (2, -1))[-1]], 4)
    self.assertEqual(d[cfunc(d, (2, -1))[2]], 1)
    d[1] = np.nan
    assert np.isnan(d[cfunc(d, (2, -1))[-1]])
    d = np.arange(47) % 7
    tgt = np.sort(np.arange(47) % 7)
    self.rnd.shuffle(d)
    for i in range(d.size):
        self.assertEqual(d[cfunc(d, i)[i]], tgt[i])
        self.argpartition_sanity_check(pyfunc, cfunc, d, i)
    d = np.array([0, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9])
    kth = [0, 3, 19, 20]
    self.assertEqual(tuple(d[cfunc(d, kth)[kth]]), (0, 3, 7, 7))
    td = [(dt, s) for dt in [np.int32, np.float32] for s in (9, 16)]
    for dt, s in td:
        d = np.arange(s, dtype=dt)
        self.rnd.shuffle(d)
        d1 = np.tile(np.arange(s, dtype=dt), (4, 1))
        map(self.rnd.shuffle, d1)
        for i in range(d.size):
            p = d[cfunc(d, i)]
            self.assertEqual(p[i], i)
            np.testing.assert_array_less(p[:i], p[i])
            np.testing.assert_array_less(p[i], p[i + 1:])
            self.argpartition_sanity_check(pyfunc, cfunc, d, i)