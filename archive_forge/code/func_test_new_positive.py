import os
import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import ChaCha20
def test_new_positive(self):
    cipher = ChaCha20.new(key=b('0') * 32, nonce=b'0' * 8)
    self.assertEqual(cipher.nonce, b'0' * 8)
    cipher = ChaCha20.new(key=b('0') * 32, nonce=b'0' * 12)
    self.assertEqual(cipher.nonce, b'0' * 12)