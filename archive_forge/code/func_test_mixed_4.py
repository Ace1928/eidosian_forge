from ctypes import *
from ctypes.test import need_symbol
from test import support
import unittest
import os
import _ctypes_test
def test_mixed_4(self):

    class X(Structure):
        _fields_ = [('a', c_short, 4), ('b', c_short, 4), ('c', c_int, 24), ('d', c_short, 4), ('e', c_short, 4), ('f', c_int, 24)]
    if os.name == 'nt':
        self.assertEqual(sizeof(X), sizeof(c_int) * 4)
    else:
        self.assertEqual(sizeof(X), sizeof(c_int) * 2)