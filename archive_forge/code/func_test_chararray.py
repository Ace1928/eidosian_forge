import unittest
from ctypes import *
def test_chararray(self):
    self.assertRaises(TypeError, delattr, (c_char * 5)(), 'value')