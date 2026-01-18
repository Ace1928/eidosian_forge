import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
def test_endian_other(self):
    self.assertIs(c_byte.__ctype_le__, c_byte)
    self.assertIs(c_byte.__ctype_be__, c_byte)
    self.assertIs(c_ubyte.__ctype_le__, c_ubyte)
    self.assertIs(c_ubyte.__ctype_be__, c_ubyte)
    self.assertIs(c_char.__ctype_le__, c_char)
    self.assertIs(c_char.__ctype_be__, c_char)