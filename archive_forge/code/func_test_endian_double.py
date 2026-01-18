import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
def test_endian_double(self):
    if sys.byteorder == 'little':
        self.assertIs(c_double.__ctype_le__, c_double)
        self.assertIs(c_double.__ctype_be__.__ctype_le__, c_double)
    else:
        self.assertIs(c_double.__ctype_be__, c_double)
        self.assertIs(c_double.__ctype_le__.__ctype_be__, c_double)
    s = c_double(math.pi)
    self.assertEqual(s.value, math.pi)
    self.assertEqual(bin(struct.pack('d', math.pi)), bin(s))
    s = c_double.__ctype_le__(math.pi)
    self.assertEqual(s.value, math.pi)
    self.assertEqual(bin(struct.pack('<d', math.pi)), bin(s))
    s = c_double.__ctype_be__(math.pi)
    self.assertEqual(s.value, math.pi)
    self.assertEqual(bin(struct.pack('>d', math.pi)), bin(s))