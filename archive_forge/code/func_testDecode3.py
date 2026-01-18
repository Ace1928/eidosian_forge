import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testDecode3(self):
    der = DerBoolean(implicit=18)
    der.decode(b'\x92\x01\x00')
    self.assertEqual(der.value, False)