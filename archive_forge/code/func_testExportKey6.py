import unittest
import re
from Cryptodome.PublicKey import DSA
from Cryptodome.SelfTest.st_common import *
from Cryptodome.Util.py3compat import *
from binascii import unhexlify
def testExportKey6(self):
    tup = (self.y, self.g, self.p, self.q, self.x)
    key = DSA.construct(tup)
    encoded = key.export_key('PEM')
    self.assertEqual(self.pem_pkcs8, encoded)
    encoded = key.export_key('PEM', pkcs8=True)
    self.assertEqual(self.pem_pkcs8, encoded)