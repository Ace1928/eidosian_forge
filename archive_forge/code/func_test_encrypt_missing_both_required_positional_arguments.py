import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_encrypt_missing_both_required_positional_arguments(self):
    """Test encrypt raises TypeError if both positional arguments missing (input and
        password)"""
    self.assertRaises(TypeError, lambda: scrypt.encrypt())