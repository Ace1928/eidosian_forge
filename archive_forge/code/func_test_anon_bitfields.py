from ctypes import *
from ctypes.test import need_symbol
from test import support
import unittest
import os
import _ctypes_test
def test_anon_bitfields(self):

    class X(Structure):
        _fields_ = [('a', c_byte, 4), ('b', c_ubyte, 4)]

    class Y(Structure):
        _anonymous_ = ['_']
        _fields_ = [('_', X)]