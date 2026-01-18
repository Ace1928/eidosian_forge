from ctypes import *
import unittest
import struct
def test_float_from_address(self):
    from array import array
    for t in float_types:
        a = array(t._type_, [3.14])
        v = t.from_address(a.buffer_info()[0])
        self.assertEqual(v.value, a[0])
        self.assertIs(type(v), t)
        a[0] = 2.3456e+17
        self.assertEqual(v.value, a[0])
        self.assertIs(type(v), t)