from __future__ import print_function
import unittest
from Cryptodome.PublicKey import RSA
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex
from Cryptodome import Random
from Cryptodome.Cipher import PKCS1_v1_5 as PKCS
from Cryptodome.Util.py3compat import b
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
def testVerify2(self):
    cipher = PKCS.new(self.key1024)
    self.assertRaises(ValueError, cipher.decrypt, '\x00' * 127, '---')
    self.assertRaises(ValueError, cipher.decrypt, '\x00' * 129, '---')
    pt = b('\x00\x02' + 'ÿ' * 7 + '\x00' + 'E' * 118)
    pt_int = bytes_to_long(pt)
    ct_int = self.key1024._encrypt(pt_int)
    ct = long_to_bytes(ct_int, 128)
    self.assertEqual(b'---', cipher.decrypt(ct, b'---'))