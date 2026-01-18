import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_int_plus(self):
    self._dll.tf_bi.restype = c_int
    self._dll.tf_bi.argtypes = (c_byte, c_int)
    self.assertEqual(self._dll.tf_bi(0, -2147483646), -715827882)
    self.assertEqual(self.S(), -2147483646)