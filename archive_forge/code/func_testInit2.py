import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testInit2(self):
    der = DerBitString(DerInteger(1))
    self.assertEqual(der.encode(), b('\x03\x04\x00\x02\x01\x01'))