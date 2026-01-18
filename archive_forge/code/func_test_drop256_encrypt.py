import unittest
from Cryptodome.Util.py3compat import b
from Cryptodome.SelfTest.st_common import list_test_cases
from binascii import unhexlify
from Cryptodome.Cipher import ARC4
def test_drop256_encrypt(self):
    cipher_drop = ARC4.new(self.key, 256)
    ct_drop = cipher_drop.encrypt(self.data[:16])
    ct = self.cipher.encrypt(self.data)[256:256 + 16]
    self.assertEqual(ct_drop, ct)