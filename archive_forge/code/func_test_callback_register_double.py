import functools
import unittest
from test import support
from ctypes import *
from ctypes.test import need_symbol
from _ctypes import CTYPES_MAX_ARGCOUNT
import _ctypes_test
def test_callback_register_double(self):
    dll = CDLL(_ctypes_test.__file__)
    CALLBACK = CFUNCTYPE(c_double, c_double, c_double, c_double, c_double, c_double)
    func = dll._testfunc_cbk_reg_double
    func.argtypes = (c_double, c_double, c_double, c_double, c_double, CALLBACK)
    func.restype = c_double

    def callback(a, b, c, d, e):
        return a + b + c + d + e
    result = func(1.1, 2.2, 3.3, 4.4, 5.5, CALLBACK(callback))
    self.assertEqual(result, callback(1.1 * 1.1, 2.2 * 2.2, 3.3 * 3.3, 4.4 * 4.4, 5.5 * 5.5))