import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testEncode7(self):
    der = DerSequence()
    der.append(384)
    der.append(b('0\x03\x02\x01\x05'))
    self.assertEqual(der.encode(), b('0\t\x02\x02\x01\x800\x03\x02\x01\x05'))
    self.assertFalse(der.hasOnlyInts())