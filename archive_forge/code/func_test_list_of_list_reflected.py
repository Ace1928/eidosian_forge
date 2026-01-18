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
@expect_reflection_failure
def test_list_of_list_reflected(self):

    def pyfunc(l1, l2):
        l1.append(l2)
        l1[-1].append(123)
    cfunc = jit(nopython=True)(pyfunc)
    l1 = [[0, 1], [2, 3]]
    l2 = [4, 5]
    expect = (list(l1), list(l2))
    got = (list(l1), list(l2))
    pyfunc(*expect)
    cfunc(*got)
    self.assertEqual(expect, got)