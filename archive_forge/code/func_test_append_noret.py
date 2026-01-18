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
def test_append_noret(self):

    def pyfunc():
        con = []
        for i in range(300):
            con.append(np.arange(i))
        c = 0.0
        for arr in con:
            c += arr.sum() / (1 + arr.size)
        return c
    cfunc = jit(nopython=True)(pyfunc)
    expect = pyfunc()
    got = cfunc()
    self.assertEqual(expect, got)