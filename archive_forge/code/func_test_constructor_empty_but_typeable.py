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
def test_constructor_empty_but_typeable(self):
    args = [np.int32(1), 10.0, 1 + 3j, [7], [17.0, 14.0], np.array([10])]
    pyfunc = list_constructor_empty_but_typeable
    for arg in args:
        cfunc = jit(nopython=True)(pyfunc)
        expected = pyfunc(arg)
        got = cfunc(arg)
        self.assertPreciseEqual(got, expected)