import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_nditer2(self):
    pyfunc = np_nditer2
    cfunc = jit(nopython=True)(pyfunc)
    for a, b in itertools.product(self.inputs(), self.inputs()):
        expected = pyfunc(a, b)
        got = cfunc(a, b)
        self.check_result(got, expected)