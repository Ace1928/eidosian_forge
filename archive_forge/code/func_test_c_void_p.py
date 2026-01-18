import unittest, sys
from ctypes import *
import _ctypes_test
def test_c_void_p(self):
    if sizeof(c_void_p) == 4:
        self.assertEqual(c_void_p(4294967295).value, c_void_p(-1).value)
        self.assertEqual(c_void_p(18446744073709551615).value, c_void_p(-1).value)
    elif sizeof(c_void_p) == 8:
        self.assertEqual(c_void_p(4294967295).value, 4294967295)
        self.assertEqual(c_void_p(18446744073709551615).value, c_void_p(-1).value)
        self.assertEqual(c_void_p(79228162514264337593543950335).value, c_void_p(-1).value)
    self.assertRaises(TypeError, c_void_p, 3.14)
    self.assertRaises(TypeError, c_void_p, object())