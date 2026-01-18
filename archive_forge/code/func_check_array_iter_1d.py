import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def check_array_iter_1d(self, arr):
    pyfunc = array_iter
    cfunc = njit((typeof(arr),))(pyfunc)
    expected = pyfunc(arr)
    self.assertPreciseEqual(cfunc(arr), expected)