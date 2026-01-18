import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
def test_endian_short(self):
    if sys.byteorder == 'little':
        self.assertIs(c_short.__ctype_le__, c_short)
        self.assertIs(c_short.__ctype_be__.__ctype_le__, c_short)
    else:
        self.assertIs(c_short.__ctype_be__, c_short)
        self.assertIs(c_short.__ctype_le__.__ctype_be__, c_short)
    s = c_short.__ctype_be__(4660)
    self.assertEqual(bin(struct.pack('>h', 4660)), '1234')
    self.assertEqual(bin(s), '1234')
    self.assertEqual(s.value, 4660)
    s = c_short.__ctype_le__(4660)
    self.assertEqual(bin(struct.pack('<h', 4660)), '3412')
    self.assertEqual(bin(s), '3412')
    self.assertEqual(s.value, 4660)
    s = c_ushort.__ctype_be__(4660)
    self.assertEqual(bin(struct.pack('>h', 4660)), '1234')
    self.assertEqual(bin(s), '1234')
    self.assertEqual(s.value, 4660)
    s = c_ushort.__ctype_le__(4660)
    self.assertEqual(bin(struct.pack('<h', 4660)), '3412')
    self.assertEqual(bin(s), '3412')
    self.assertEqual(s.value, 4660)