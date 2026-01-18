import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testDecode9(self):
    der = DerSequence()
    self.assertEqual(der, der.decode(b('0\x06$\x02¶c\x12\x00')))