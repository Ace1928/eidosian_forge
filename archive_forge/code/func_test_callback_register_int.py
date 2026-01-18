import functools
import unittest
from test import support
from ctypes import *
from ctypes.test import need_symbol
from _ctypes import CTYPES_MAX_ARGCOUNT
import _ctypes_test
def test_callback_register_int(self):
    dll = CDLL(_ctypes_test.__file__)
    CALLBACK = CFUNCTYPE(c_int, c_int, c_int, c_int, c_int, c_int)
    func = dll._testfunc_cbk_reg_int
    func.argtypes = (c_int, c_int, c_int, c_int, c_int, CALLBACK)
    func.restype = c_int

    def callback(a, b, c, d, e):
        return a + b + c + d + e
    result = func(2, 3, 4, 5, 6, CALLBACK(callback))
    self.assertEqual(result, callback(2 * 2, 3 * 3, 4 * 4, 5 * 5, 6 * 6))