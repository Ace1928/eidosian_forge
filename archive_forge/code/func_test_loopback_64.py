import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
from Cryptodome.Cipher import DES, DES3, ARC2, CAST, Blowfish
def test_loopback_64(self):
    cipher = DES3.new(self.key_192, DES3.MODE_EAX, nonce=self.nonce_96)
    pt = get_tag_random('plaintext', 8 * 100)
    ct = cipher.encrypt(pt)
    cipher = DES3.new(self.key_192, DES3.MODE_EAX, nonce=self.nonce_96)
    pt2 = cipher.decrypt(ct)
    self.assertEqual(pt, pt2)