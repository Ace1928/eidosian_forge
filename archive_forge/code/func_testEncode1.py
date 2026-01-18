import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testEncode1(self):
    der = DerBoolean(False)
    self.assertEqual(der.encode(), b'\x01\x01\x00')