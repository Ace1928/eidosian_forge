import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testObjEncode2(self):
    der = DerObject(3, b('\x12\x12'))
    self.assertEqual(der.encode(), b('\x03\x02\x12\x12'))