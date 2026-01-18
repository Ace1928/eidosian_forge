import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testObjEncode3(self):
    der = DerObject(b('\x10'))
    der.payload = b('0') * 128
    self.assertEqual(der.encode(), b('\x10\x81\x80' + '0' * 128))