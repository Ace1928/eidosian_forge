from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
@need_symbol('c_longdouble')
def test_longdoubleresult(self):
    f = dll._testfunc_D_bhilfD
    f.argtypes = [c_byte, c_short, c_int, c_long, c_float, c_longdouble]
    f.restype = c_longdouble
    result = f(1, 2, 3, 4, 5.0, 6.0)
    self.assertEqual(result, 21)
    self.assertEqual(type(result), float)
    result = f(-1, -2, -3, -4, -5.0, -6.0)
    self.assertEqual(result, -21)
    self.assertEqual(type(result), float)