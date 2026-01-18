import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testObjDecode1(self):
    der = DerObject(2)
    der.decode(b('\x02\x02\x01\x02'))
    self.assertEqual(der.payload, b('\x01\x02'))
    self.assertEqual(der._tag_octet, 2)