import unittest
import sys
import numpy
from numba import jit, njit
from numba.core import types, utils
from numba.tests.support import tag
from numba.cpython.rangeobj import length_of_iterator
def test_loop2_int16(self):
    pyfunc = loop2
    cfunc = njit((types.int16, types.int16))(pyfunc)
    self.assertTrue(cfunc(1, 6), pyfunc(1, 6))