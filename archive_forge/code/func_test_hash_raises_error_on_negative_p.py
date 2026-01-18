import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_hash_raises_error_on_negative_p(self):
    """Test hash raises scrypt error on illegal parameter value (p < 0)"""
    self.assertRaises(scrypt.error, lambda: scrypt.hash(self.input, self.salt, p=-1))