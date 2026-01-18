import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_nditer3(self):
    pyfunc = np_nditer3
    cfunc = jit(nopython=True)(pyfunc)
    inputs = self.basic_inputs
    for a, b, c in itertools.product(inputs(), inputs(), inputs()):
        expected = pyfunc(a, b, c)
        got = cfunc(a, b, c)
        self.check_result(got, expected)