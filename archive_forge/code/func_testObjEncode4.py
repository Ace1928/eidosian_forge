import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def testObjEncode4(self):
    der = DerObject(16, implicit=1, constructed=True)
    der.payload = b('ppll')
    self.assertEqual(der.encode(), b('¡\x04ppll'))
    der = DerObject(2, implicit=30, constructed=False)
    der.payload = b('ppll')
    self.assertEqual(der.encode(), b('\x9e\x04ppll'))