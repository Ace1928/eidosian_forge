import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_hash_buflen_keyword(self):
    """Test hash takes keyword valid buflen."""
    h64 = scrypt.hash(self.input, self.salt, buflen=64)
    h128 = scrypt.hash(self.input, self.salt, buflen=128)
    self.assertEqual(len(h64), 64)
    self.assertEqual(len(h128), 128)