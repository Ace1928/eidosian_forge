import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_invalid_encrypt_or_update_after_digest(self):
    for method_name in ('encrypt', 'update'):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.encrypt(self.data)
        cipher.encrypt()
        cipher.digest()
        self.assertRaises(TypeError, getattr(cipher, method_name), self.data)
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.encrypt_and_digest(self.data)