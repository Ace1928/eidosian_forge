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
def test_c06(self):

    def bar(x):
        f = x
        f[0][0] = np.array([x + 1j for x in np.arange(10)])
        return f
    r = [[np.arange(3)]]
    with self.assertRaises(errors.TypingError) as raises:
        self.compile_and_test(bar, r)
    self.assertIn('invalid setitem with value', str(raises.exception))