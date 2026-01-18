from ctypes import *
from ctypes.test import need_symbol
from test import support
import unittest
import os
import _ctypes_test
def test_multi_bitfields_size(self):

    class X(Structure):
        _fields_ = [('a', c_short, 1), ('b', c_short, 14), ('c', c_short, 1)]
    self.assertEqual(sizeof(X), sizeof(c_short))

    class X(Structure):
        _fields_ = [('a', c_short, 1), ('a1', c_short), ('b', c_short, 14), ('c', c_short, 1)]
    self.assertEqual(sizeof(X), sizeof(c_short) * 3)
    self.assertEqual(X.a.offset, 0)
    self.assertEqual(X.a1.offset, sizeof(c_short))
    self.assertEqual(X.b.offset, sizeof(c_short) * 2)
    self.assertEqual(X.c.offset, sizeof(c_short) * 2)

    class X(Structure):
        _fields_ = [('a', c_short, 3), ('b', c_short, 14), ('c', c_short, 14)]
    self.assertEqual(sizeof(X), sizeof(c_short) * 3)
    self.assertEqual(X.a.offset, sizeof(c_short) * 0)
    self.assertEqual(X.b.offset, sizeof(c_short) * 1)
    self.assertEqual(X.c.offset, sizeof(c_short) * 2)