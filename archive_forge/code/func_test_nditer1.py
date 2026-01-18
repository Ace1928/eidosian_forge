import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_nditer1(self):
    pyfunc = np_nditer1
    cfunc = jit(nopython=True)(pyfunc)
    for a in self.inputs():
        expected = pyfunc(a)
        got = cfunc(a)
        self.check_result(got, expected)