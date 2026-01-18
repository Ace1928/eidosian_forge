from ctypes import *
import unittest
def test_64(self):
    self.assertEqual(8, sizeof(c_int64))
    self.assertEqual(8, sizeof(c_uint64))