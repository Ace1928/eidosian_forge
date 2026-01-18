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
def test_list_of_list_ctor(self):

    @njit
    def bar(x):
        pass

    @njit
    def foo():
        x = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]]
        bar(x)
    foo()
    larg = bar.signatures[0][0]
    self.assertEqual(larg.initial_value, None)
    self.assertEqual(larg.dtype.initial_value, None)