import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_OAEP as PKCS
from Cryptodome.Hash import MD2, MD5, SHA1, SHA256, RIPEMD160, SHA224, SHA384, SHA512
from Cryptodome import Random
from Cryptodome.Signature.pss import MGF1
from Cryptodome.Util.py3compat import b, bchr
def testEncryptDecrypt1(self):
    for pt_len in range(0, 128 - 2 * 20 - 2):
        pt = self.rng(pt_len)
        cipher = PKCS.new(self.key1024)
        ct = cipher.encrypt(pt)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)