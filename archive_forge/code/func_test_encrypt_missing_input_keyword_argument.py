import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_encrypt_missing_input_keyword_argument(self):
    """Test encrypt raises TypeError if keyword argument missing input."""
    self.assertRaises(TypeError, lambda: scrypt.encrypt(password=self.password))