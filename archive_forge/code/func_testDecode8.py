import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testDecode8(self):
    der = DerSequence()
    der.decode(b('0\x06$\x02¶c\x12\x00'))
    self.assertEqual(len(der), 2)
    self.assertEqual(der[0], b('$\x02¶c'))
    self.assertEqual(der[1], b('\x12\x00'))
    self.assertEqual(der.hasInts(), 0)
    self.assertEqual(der.hasInts(False), 0)
    self.assertFalse(der.hasOnlyInts())
    self.assertFalse(der.hasOnlyInts(False))