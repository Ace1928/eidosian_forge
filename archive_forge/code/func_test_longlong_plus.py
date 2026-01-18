import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
@need_symbol('c_longlong')
def test_longlong_plus(self):
    self._dll.tf_bq.restype = c_longlong
    self._dll.tf_bq.argtypes = (c_byte, c_longlong)
    self.assertEqual(self._dll.tf_bq(0, -9223372036854775806), -3074457345618258602)
    self.assertEqual(self.S(), -9223372036854775806)