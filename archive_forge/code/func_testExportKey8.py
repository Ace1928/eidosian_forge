import unittest
import re
from Cryptodome.PublicKey import DSA
from Cryptodome.SelfTest.st_common import *
from Cryptodome.Util.py3compat import *
from binascii import unhexlify
def testExportKey8(self):
    tup = (self.y, self.g, self.p, self.q, self.x)
    key = DSA.construct(tup)
    encoded = key.export_key('PEM', pkcs8=False, passphrase='PWDTEST')
    key = DSA.importKey(encoded, 'PWDTEST')
    self.assertEqual(self.y, key.y)
    self.assertEqual(self.p, key.p)
    self.assertEqual(self.q, key.q)
    self.assertEqual(self.g, key.g)
    self.assertEqual(self.x, key.x)