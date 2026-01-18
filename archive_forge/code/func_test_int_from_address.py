from ctypes import *
import unittest
import struct
def test_int_from_address(self):
    from array import array
    for t in signed_types + unsigned_types:
        try:
            array(t._type_)
        except ValueError:
            continue
        a = array(t._type_, [100])
        v = t.from_address(a.buffer_info()[0])
        self.assertEqual(v.value, a[0])
        self.assertEqual(type(v), t)
        a[0] = 42
        self.assertEqual(v.value, a[0])