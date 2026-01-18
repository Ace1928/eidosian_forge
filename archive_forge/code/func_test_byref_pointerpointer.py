import unittest
from ctypes.test import need_symbol
import test.support
def test_byref_pointerpointer(self):
    from ctypes import c_short, c_uint, c_int, c_long, pointer, POINTER, byref
    LPLPINT = POINTER(POINTER(c_int))
    LPLPINT.from_param(byref(pointer(c_int(42))))
    self.assertRaises(TypeError, LPLPINT.from_param, byref(pointer(c_short(22))))
    if c_int != c_long:
        self.assertRaises(TypeError, LPLPINT.from_param, byref(pointer(c_long(22))))
    self.assertRaises(TypeError, LPLPINT.from_param, byref(pointer(c_uint(22))))