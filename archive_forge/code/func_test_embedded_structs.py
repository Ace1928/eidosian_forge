import unittest
from ctypes import *
from sys import getrefcount as grc
def test_embedded_structs(self):

    class X(Structure):
        _fields_ = [('a', c_int), ('b', c_int)]

    class Y(Structure):
        _fields_ = [('x', X), ('y', X)]
    y = Y()
    self.assertEqual(y._objects, None)
    x1, x2 = (X(), X())
    y.x, y.y = (x1, x2)
    self.assertEqual(y._objects, {'0': {}, '1': {}})
    x1.a, x2.b = (42, 93)
    self.assertEqual(y._objects, {'0': {}, '1': {}})