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
def test_c07(self):
    self.disable_leak_check()

    def bar(x):
        return x[-7]
    r = [[np.arange(3)]]
    cfunc = jit(nopython=True)(bar)
    with self.assertRaises(IndexError) as raises:
        cfunc(r)
    self.assertIn('getitem out of range', str(raises.exception))