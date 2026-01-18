import unittest
import ctypes
from ctypes.test import need_symbol
import _ctypes_test
def test_embedded_null(self):

    class TestStruct(ctypes.Structure):
        _fields_ = [('unicode', ctypes.c_wchar_p)]
    t = TestStruct()
    t.unicode = 'foo\x00bar\x00\x00'