import unittest
from ctypes import *
import _ctypes_test
def test_from_dll_refcount(self):

    class BadSequence(tuple):

        def __getitem__(self, key):
            if key == 0:
                return 'my_strchr'
            if key == 1:
                return CDLL(_ctypes_test.__file__)
            raise IndexError
    strchr = CFUNCTYPE(c_char_p, c_char_p, c_char)(BadSequence(('my_strchr', CDLL(_ctypes_test.__file__))))
    self.assertTrue(strchr(b'abcdef', b'b'), 'bcdef')
    self.assertEqual(strchr(b'abcdef', b'x'), None)
    self.assertRaises(ArgumentError, strchr, b'abcdef', 3.0)
    self.assertRaises(TypeError, strchr, b'abcdef')