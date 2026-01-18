from ctypes import *
from ctypes.test import need_symbol
import unittest
import _ctypes_test
def test_c_wchar_p_arg(self):
    func = testdll._testfunc_p_p
    func.restype = c_wchar_p
    func.argtypes = (c_wchar_p,)
    c_wchar_p.from_param('123')
    self.assertEqual(None, func(None))
    self.assertEqual('123', func('123'))
    self.assertEqual(None, func(c_wchar_p(None)))
    self.assertEqual('123', func(c_wchar_p('123')))
    self.assertEqual('123', func(c_wbuffer('123')))
    ca = c_wchar('a')
    self.assertEqual('a', func(pointer(ca))[0])
    self.assertEqual('a', func(byref(ca))[0])