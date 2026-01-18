import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testObjEncode1(self):
    der = DerObject(b('\x02'))
    self.assertEqual(der.encode(), b('\x02\x00'))
    der.payload = b('E')
    self.assertEqual(der.encode(), b('\x02\x01E'))
    self.assertEqual(der.encode(), b('\x02\x01E'))
    der = DerObject(4)
    der.payload = b('E')
    self.assertEqual(der.encode(), b('\x04\x01E'))
    der = DerObject(b('\x10'), constructed=True)
    self.assertEqual(der.encode(), b('0\x00'))