import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_encrypt_maxmemfrac_positional(self):
    """Test encrypt maxmemfrac accepts 5th positional argument of 1/16 total memory
        for V array."""
    s = scrypt.encrypt(self.input, self.password, 0.01, 0, 0.0625)
    m = scrypt.decrypt(s, self.password)
    self.assertEqual(m, self.input)