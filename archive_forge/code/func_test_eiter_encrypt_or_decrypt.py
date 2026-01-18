import os
import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import ChaCha20
def test_eiter_encrypt_or_decrypt(self):
    """Verify that a cipher cannot be used for both decrypting and encrypting"""
    c1 = ChaCha20.new(key=b('5') * 32, nonce=b('6') * 8)
    c1.encrypt(b('8'))
    self.assertRaises(TypeError, c1.decrypt, b('9'))
    c2 = ChaCha20.new(key=b('5') * 32, nonce=b('6') * 8)
    c2.decrypt(b('8'))
    self.assertRaises(TypeError, c2.encrypt, b('9'))