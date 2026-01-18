import os
import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import ChaCha20
def test_new_negative(self):
    new = ChaCha20.new
    self.assertRaises(TypeError, new)
    self.assertRaises(TypeError, new, nonce=b('0'))
    self.assertRaises(ValueError, new, nonce=b('0') * 8, key=b('0'))
    self.assertRaises(ValueError, new, nonce=b('0'), key=b('0') * 32)