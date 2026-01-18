import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testErrDecode2(self):
    der = DerSequence()
    self.assertRaises(ValueError, der.decode, b('0\x00\x00'))