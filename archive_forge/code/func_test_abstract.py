import unittest, sys
from ctypes import *
import _ctypes_test
def test_abstract(self):
    from ctypes import _Pointer
    self.assertRaises(TypeError, _Pointer.set_type, 42)