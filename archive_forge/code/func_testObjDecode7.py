import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testObjDecode7(self):
    der = DerObject(16, explicit=5)
    der.decode(b('¥\x06\x10\x04xxll'))
    self.assertEqual(der._inner_tag_octet, 16)
    self.assertEqual(der.payload, b('xxll'))
    der = DerObject(16, explicit=0)
    der.decode(b('\xa0\x06\x10\x04xxll'))
    self.assertEqual(der._inner_tag_octet, 16)
    self.assertEqual(der.payload, b('xxll'))