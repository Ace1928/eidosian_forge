import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
def test_endian_int(self):
    if sys.byteorder == 'little':
        self.assertIs(c_int.__ctype_le__, c_int)
        self.assertIs(c_int.__ctype_be__.__ctype_le__, c_int)
    else:
        self.assertIs(c_int.__ctype_be__, c_int)
        self.assertIs(c_int.__ctype_le__.__ctype_be__, c_int)
    s = c_int.__ctype_be__(305419896)
    self.assertEqual(bin(struct.pack('>i', 305419896)), '12345678')
    self.assertEqual(bin(s), '12345678')
    self.assertEqual(s.value, 305419896)
    s = c_int.__ctype_le__(305419896)
    self.assertEqual(bin(struct.pack('<i', 305419896)), '78563412')
    self.assertEqual(bin(s), '78563412')
    self.assertEqual(s.value, 305419896)
    s = c_uint.__ctype_be__(305419896)
    self.assertEqual(bin(struct.pack('>I', 305419896)), '12345678')
    self.assertEqual(bin(s), '12345678')
    self.assertEqual(s.value, 305419896)
    s = c_uint.__ctype_le__(305419896)
    self.assertEqual(bin(struct.pack('<I', 305419896)), '78563412')
    self.assertEqual(bin(s), '78563412')
    self.assertEqual(s.value, 305419896)