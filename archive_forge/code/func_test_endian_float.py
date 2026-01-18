import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
def test_endian_float(self):
    if sys.byteorder == 'little':
        self.assertIs(c_float.__ctype_le__, c_float)
        self.assertIs(c_float.__ctype_be__.__ctype_le__, c_float)
    else:
        self.assertIs(c_float.__ctype_be__, c_float)
        self.assertIs(c_float.__ctype_le__.__ctype_be__, c_float)
    s = c_float(math.pi)
    self.assertEqual(bin(struct.pack('f', math.pi)), bin(s))
    self.assertAlmostEqual(s.value, math.pi, places=6)
    s = c_float.__ctype_le__(math.pi)
    self.assertAlmostEqual(s.value, math.pi, places=6)
    self.assertEqual(bin(struct.pack('<f', math.pi)), bin(s))
    s = c_float.__ctype_be__(math.pi)
    self.assertAlmostEqual(s.value, math.pi, places=6)
    self.assertEqual(bin(struct.pack('>f', math.pi)), bin(s))