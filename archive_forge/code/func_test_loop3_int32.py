import unittest
import sys
import numpy
from numba import jit, njit
from numba.core import types, utils
from numba.tests.support import tag
from numba.cpython.rangeobj import length_of_iterator
def test_loop3_int32(self):
    pyfunc = loop3
    cfunc = njit((types.int32, types.int32, types.int32))(pyfunc)
    arglist = [(1, 2, 1), (2, 8, 3), (-10, -11, -10), (-10, -10, -2)]
    for args in arglist:
        self.assertEqual(cfunc(*args), pyfunc(*args))