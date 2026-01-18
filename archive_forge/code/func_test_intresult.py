from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
def test_intresult(self):
    f = dll._testfunc_i_bhilfd
    f.argtypes = [c_byte, c_short, c_int, c_long, c_float, c_double]
    f.restype = c_int
    result = f(1, 2, 3, 4, 5.0, 6.0)
    self.assertEqual(result, 21)
    self.assertEqual(type(result), int)
    result = f(-1, -2, -3, -4, -5.0, -6.0)
    self.assertEqual(result, -21)
    self.assertEqual(type(result), int)
    f.restype = c_short
    result = f(1, 2, 3, 4, 5.0, 6.0)
    self.assertEqual(result, 21)
    self.assertEqual(type(result), int)
    result = f(1, 2, 3, 65540, 5.0, 6.0)
    self.assertEqual(result, 21)
    self.assertEqual(type(result), int)
    self.assertRaises(TypeError, setattr, f, 'restype', 'i')