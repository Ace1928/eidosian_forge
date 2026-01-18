from ctypes import *
from ctypes.test import need_symbol
from test import support
import unittest
import os
import _ctypes_test
def test_mixed_1(self):

    class X(Structure):
        _fields_ = [('a', c_byte, 4), ('b', c_int, 4)]
    if os.name == 'nt':
        self.assertEqual(sizeof(X), sizeof(c_int) * 2)
    else:
        self.assertEqual(sizeof(X), sizeof(c_int))