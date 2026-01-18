import unittest, sys
from ctypes import *
import _ctypes_test
def test_charpp(self):
    """Test that a character pointer-to-pointer is correctly passed"""
    dll = CDLL(_ctypes_test.__file__)
    func = dll._testfunc_c_p_p
    func.restype = c_char_p
    argv = (c_char_p * 2)()
    argc = c_int(2)
    argv[0] = b'hello'
    argv[1] = b'world'
    result = func(byref(argc), argv)
    self.assertEqual(result, b'world')