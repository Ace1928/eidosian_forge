import json
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.Util.py3compat import tobytes, bchr
from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHAKE128
from Cryptodome.Util.strxor import strxor
def test_invalid_init_decrypt(self):
    cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
    self.assertRaises(TypeError, cipher.decrypt, b'xxx')