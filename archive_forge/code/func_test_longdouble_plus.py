import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
@need_symbol('c_longdouble')
def test_longdouble_plus(self):
    self._dll.tf_bD.restype = c_longdouble
    self._dll.tf_bD.argtypes = (c_byte, c_longdouble)
    self.assertEqual(self._dll.tf_bD(0, 42.0), 14.0)
    self.assertEqual(self.S(), 42)