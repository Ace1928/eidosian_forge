import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_hash_p_keyword(self):
    """Test hash takes keyword valid p."""
    h = scrypt.hash(p=4, password=self.input, salt=self.salt)
    self.assertEqual(len(h), 64)