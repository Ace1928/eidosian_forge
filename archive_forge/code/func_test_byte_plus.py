import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_byte_plus(self):
    self._dll.tf_bb.restype = c_byte
    self._dll.tf_bb.argtypes = (c_byte, c_byte)
    self.assertEqual(self._dll.tf_bb(0, -126), -42)
    self.assertEqual(self.S(), -126)