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
def test_c01(self):

    def bar(x):
        return x.pop()
    r = [[np.zeros(0)], [np.zeros(10) * 1j]]
    self.compile_and_test(bar, r)
    with self.assertRaises(TypeError) as raises:
        self.compile_and_test(bar, r)
    self.assertIn('reflected list(array(float64, 1d, C)) != reflected list(array(complex128, 1d, C))', str(raises.exception))