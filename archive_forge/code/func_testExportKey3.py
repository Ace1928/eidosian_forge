import unittest
import re
from Cryptodome.PublicKey import DSA
from Cryptodome.SelfTest.st_common import *
from Cryptodome.Util.py3compat import *
from binascii import unhexlify
def testExportKey3(self):
    tup = (self.y, self.g, self.p, self.q, self.x)
    key = DSA.construct(tup)
    encoded = key.export_key('DER', pkcs8=False)
    self.assertEqual(self.der_private, encoded)