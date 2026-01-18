import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testDecode2(self):
    der = DerBoolean()
    der.decode(b'\x01\x01\xff')
    self.assertEqual(der.value, True)