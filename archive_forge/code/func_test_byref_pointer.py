import unittest
from ctypes.test import need_symbol
import test.support
def test_byref_pointer(self):
    from ctypes import c_short, c_uint, c_int, c_long, POINTER, byref
    LPINT = POINTER(c_int)
    LPINT.from_param(byref(c_int(42)))
    self.assertRaises(TypeError, LPINT.from_param, byref(c_short(22)))
    if c_int != c_long:
        self.assertRaises(TypeError, LPINT.from_param, byref(c_long(22)))
    self.assertRaises(TypeError, LPINT.from_param, byref(c_uint(22)))