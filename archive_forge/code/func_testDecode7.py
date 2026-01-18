import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testDecode7(self):
    der = DerSequence()
    der.decode(b('0\n\x02\x02\x01\x80$\x02¶c\x12\x00'))
    self.assertEqual(len(der), 3)
    self.assertEqual(der[0], 384)
    self.assertEqual(der[1], b('$\x02¶c'))
    self.assertEqual(der[2], b('\x12\x00'))