from ctypes import *
import unittest
import struct
def test_char_from_address(self):
    from ctypes import c_char
    from array import array
    a = array('b', [0])
    a[0] = ord('x')
    v = c_char.from_address(a.buffer_info()[0])
    self.assertEqual(v.value, b'x')
    self.assertIs(type(v), c_char)
    a[0] = ord('?')
    self.assertEqual(v.value, b'?')