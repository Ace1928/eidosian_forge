import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_hash_p_positional(self):
    """Test hash accepts valid p in position 5."""
    h = scrypt.hash(self.input, self.salt, 256, 8, 2)
    self.assertEqual(len(h), 64)