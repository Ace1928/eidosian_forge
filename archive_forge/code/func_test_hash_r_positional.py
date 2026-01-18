import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_hash_r_positional(self):
    """Test hash accepts valid r in position 4."""
    h = scrypt.hash(self.input, self.salt, 256, 16)
    self.assertEqual(len(h), 64)