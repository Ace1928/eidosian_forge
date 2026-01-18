import unittest
from ctypes.test import need_symbol
import test.support
@need_symbol('c_wchar_p')
def test_cw_strings(self):
    from ctypes import c_wchar_p
    c_wchar_p.from_param('123')
    self.assertRaises(TypeError, c_wchar_p.from_param, 42)
    self.assertRaises(TypeError, c_wchar_p.from_param, b'123\xff')
    pa = c_wchar_p.from_param(c_wchar_p('123'))
    self.assertEqual(type(pa), c_wchar_p)