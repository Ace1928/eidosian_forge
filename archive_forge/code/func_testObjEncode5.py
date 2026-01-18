import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testObjEncode5(self):
    der = DerObject(16, explicit=5)
    der.payload = b('xxll')
    self.assertEqual(der.encode(), b('¥\x06\x10\x04xxll'))