import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_hash_raises_error_n_under_limit(self):
    """Test hash raises scrypt error when parameter N under limit of 1."""
    self.assertRaises(scrypt.error, lambda: scrypt.hash(self.input, self.salt, N=1))
    self.assertRaises(scrypt.error, lambda: scrypt.hash(self.input, self.salt, N=-1))