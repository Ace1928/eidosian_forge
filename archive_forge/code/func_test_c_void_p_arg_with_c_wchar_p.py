from ctypes import *
from ctypes.test import need_symbol
import unittest
import _ctypes_test
@need_symbol('c_wchar_p')
def test_c_void_p_arg_with_c_wchar_p(self):
    func = testdll._testfunc_p_p
    func.restype = c_wchar_p
    func.argtypes = (c_void_p,)
    self.assertEqual(None, func(c_wchar_p(None)))
    self.assertEqual('123', func(c_wchar_p('123')))