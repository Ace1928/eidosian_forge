from ctypes import *
from ctypes.test import need_symbol
import unittest
import _ctypes_test
def test_int_pointer_arg(self):
    func = testdll._testfunc_p_p
    if sizeof(c_longlong) == sizeof(c_void_p):
        func.restype = c_longlong
    else:
        func.restype = c_long
    self.assertEqual(0, func(0))
    ci = c_int(0)
    func.argtypes = (POINTER(c_int),)
    self.assertEqual(positive_address(addressof(ci)), positive_address(func(byref(ci))))
    func.argtypes = (c_char_p,)
    self.assertRaises(ArgumentError, func, byref(ci))
    func.argtypes = (POINTER(c_short),)
    self.assertRaises(ArgumentError, func, byref(ci))
    func.argtypes = (POINTER(c_double),)
    self.assertRaises(ArgumentError, func, byref(ci))