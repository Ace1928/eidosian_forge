from ctypes import *
import unittest
def test_ccharp(self):
    x = c_char_p()
    self.assertEqual(x._objects, None)
    x.value = b'abc'
    self.assertEqual(x._objects, b'abc')
    x = c_char_p(b'spam')
    self.assertEqual(x._objects, b'spam')