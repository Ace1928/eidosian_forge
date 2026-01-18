import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_invalid_init_encrypt_decrypt_digest_verify(self):
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    ct = cipher.encrypt(self.data)
    self.assertRaises(TypeError, cipher.digest)
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    cipher.decrypt(ct)
    self.assertRaises(TypeError, cipher.verify)