import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_char_array(self):
    s = b'abcdefghijklmnopqrstuvwxyz\x00'
    p = (c_char * 27)(*s)
    self.assertEqual(p[:], s)
    self.assertEqual(p[:], s)
    self.assertEqual(p[::-1], s[::-1])
    self.assertEqual(p[5::-2], s[5::-2])
    self.assertEqual(p[2:5:-3], s[2:5:-3])