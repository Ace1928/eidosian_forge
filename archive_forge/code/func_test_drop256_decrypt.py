import unittest
from Cryptodome.Util.py3compat import b
from Cryptodome.SelfTest.st_common import list_test_cases
from binascii import unhexlify
from Cryptodome.Cipher import ARC4
def test_drop256_decrypt(self):
    cipher_drop = ARC4.new(self.key, 256)
    pt_drop = cipher_drop.decrypt(self.data[:16])
    pt = self.cipher.decrypt(self.data)[256:256 + 16]
    self.assertEqual(pt_drop, pt)