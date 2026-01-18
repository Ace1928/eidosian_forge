import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testDecode5(self):
    der = DerInteger()
    der.decode(b('\x02\x02\x00\x01'))
    self.assertEqual(der.value, 1)
    der.decode(b('\x02\x02ÿÿ'))
    self.assertEqual(der.value, -1)
    der.decode(b('\x02\x00'))
    self.assertEqual(der.value, 0)