import os
import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import ChaCha20
def test_default_nonce(self):
    cipher1 = ChaCha20.new(key=bchr(1) * 32)
    cipher2 = ChaCha20.new(key=bchr(1) * 32)
    self.assertEqual(len(cipher1.nonce), 8)
    self.assertNotEqual(cipher1.nonce, cipher2.nonce)