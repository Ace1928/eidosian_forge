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
def test_c05(self):

    def bar(x):
        f = x
        f[0][0] = np.array([x for x in np.arange(10).astype(np.intp)])
        return f
    r = [[np.arange(3).astype(np.intp)]]
    self.compile_and_test(bar, r)