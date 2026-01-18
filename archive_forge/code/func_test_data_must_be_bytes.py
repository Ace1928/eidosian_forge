import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.number import long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
def test_data_must_be_bytes(self):
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    self.assertRaises(TypeError, cipher.encrypt, u'test1234567890-*')
    cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
    self.assertRaises(TypeError, cipher.decrypt, u'test1234567890-*')