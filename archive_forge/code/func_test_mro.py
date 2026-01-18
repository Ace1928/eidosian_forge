from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
def test_mro(self):
    with self.assertRaises(TypeError):

        class X(object, Array):
            _length_ = 5
            _type_ = 'i'
    from _ctypes import _Pointer
    with self.assertRaises(TypeError):

        class X2(object, _Pointer):
            pass
    from _ctypes import _SimpleCData
    with self.assertRaises(TypeError):

        class X3(object, _SimpleCData):
            _type_ = 'i'
    with self.assertRaises(TypeError):

        class X4(object, Structure):
            _fields_ = []