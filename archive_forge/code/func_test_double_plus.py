import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_double_plus(self):
    self._dll.tf_bd.restype = c_double
    self._dll.tf_bd.argtypes = (c_byte, c_double)
    self.assertEqual(self._dll.tf_bd(0, 42.0), 14.0)
    self.assertEqual(self.S(), 42)