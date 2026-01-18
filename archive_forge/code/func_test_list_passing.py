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
def test_list_passing(self):

    @jit(nopython=True)
    def inner(lst):
        return (len(lst), lst[-1])

    @jit(nopython=True)
    def outer(n):
        l = list(range(n))
        return inner(l)
    self.assertPreciseEqual(outer(5), (5, 4))