import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
def test_longer_assoc_data_than_declared(self):
    cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96, assoc_len=0)
    self.assertRaises(ValueError, cipher.update, b'1')
    cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96, assoc_len=15)
    self.assertRaises(ValueError, cipher.update, self.data)