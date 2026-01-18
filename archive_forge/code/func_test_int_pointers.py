import unittest
from ctypes.test import need_symbol
import test.support
def test_int_pointers(self):
    from ctypes import c_short, c_uint, c_int, c_long, POINTER, pointer
    LPINT = POINTER(c_int)
    x = LPINT.from_param(pointer(c_int(42)))
    self.assertEqual(x.contents.value, 42)
    self.assertEqual(LPINT(c_int(42)).contents.value, 42)
    self.assertEqual(LPINT.from_param(None), None)
    if c_int != c_long:
        self.assertRaises(TypeError, LPINT.from_param, pointer(c_long(42)))
    self.assertRaises(TypeError, LPINT.from_param, pointer(c_uint(42)))
    self.assertRaises(TypeError, LPINT.from_param, pointer(c_short(42)))