import unittest
from ctypes import *
import _ctypes_test
def test_dllfunctions(self):

    def NoNullHandle(value):
        if not value:
            raise WinError()
        return value
    strchr = lib.my_strchr
    strchr.restype = c_char_p
    strchr.argtypes = (c_char_p, c_char)
    self.assertEqual(strchr(b'abcdefghi', b'b'), b'bcdefghi')
    self.assertEqual(strchr(b'abcdefghi', b'x'), None)
    strtok = lib.my_strtok
    strtok.restype = c_char_p

    def c_string(init):
        size = len(init) + 1
        return (c_char * size)(*init)
    s = b'a\nb\nc'
    b = c_string(s)
    self.assertEqual(strtok(b, b'\n'), b'a')
    self.assertEqual(strtok(None, b'\n'), b'b')
    self.assertEqual(strtok(None, b'\n'), b'c')
    self.assertEqual(strtok(None, b'\n'), None)