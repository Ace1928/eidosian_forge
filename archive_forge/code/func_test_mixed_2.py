from ctypes import *
from ctypes.test import need_symbol
from test import support
import unittest
import os
import _ctypes_test
def test_mixed_2(self):

    class X(Structure):
        _fields_ = [('a', c_byte, 4), ('b', c_int, 32)]
    self.assertEqual(sizeof(X), alignment(c_int) + sizeof(c_int))