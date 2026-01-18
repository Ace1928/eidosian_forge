import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def check_type_constructor(self, np_type, values):
    pyfunc = converter(np_type)
    cfunc = jit(nopython=True)(pyfunc)
    for val in values:
        expected = np_type(val)
        got = cfunc(val)
        self.assertPreciseEqual(got, expected)