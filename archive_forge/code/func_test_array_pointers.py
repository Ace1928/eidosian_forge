import unittest
from ctypes.test import need_symbol
import test.support
def test_array_pointers(self):
    from ctypes import c_short, c_uint, c_int, c_long, POINTER
    INTARRAY = c_int * 3
    ia = INTARRAY()
    self.assertEqual(len(ia), 3)
    self.assertEqual([ia[i] for i in range(3)], [0, 0, 0])
    LPINT = POINTER(c_int)
    LPINT.from_param((c_int * 3)())
    self.assertRaises(TypeError, LPINT.from_param, c_short * 3)
    self.assertRaises(TypeError, LPINT.from_param, c_long * 3)
    self.assertRaises(TypeError, LPINT.from_param, c_uint * 3)