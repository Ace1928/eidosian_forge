from ctypes import *
import unittest
def test_array_invalid_length(self):
    self.assertRaises(ValueError, lambda: c_int * -1)
    self.assertRaises(ValueError, lambda: c_int * -3)