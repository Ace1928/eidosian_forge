import unittest
import re
from Cryptodome.PublicKey import DSA
from Cryptodome.SelfTest.st_common import *
from Cryptodome.Util.py3compat import *
from binascii import unhexlify
def test_exportKey(self):
    tup = (self.y, self.g, self.p, self.q, self.x)
    key = DSA.construct(tup)
    self.assertEqual(key.exportKey(), key.export_key())