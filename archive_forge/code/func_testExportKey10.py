import unittest
import re
from Cryptodome.PublicKey import DSA
from Cryptodome.SelfTest.st_common import *
from Cryptodome.Util.py3compat import *
from binascii import unhexlify
def testExportKey10(self):
    tup = (self.y, self.g, self.p, self.q, self.x)
    key = DSA.construct(tup)
    randfunc = BytesIO(unhexlify(b('27A1C66C42AFEECE') + b('D725BF1B6B8239F4'))).read
    encoded = key.export_key('DER', pkcs8=True, passphrase='PWDTEST', randfunc=randfunc)
    self.assertEqual(self.der_pkcs8_encrypted, encoded)