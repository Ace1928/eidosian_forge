import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_decrypt_maxmemfrac_positional(self):
    """Test decrypt function accepts maxmem keyword argument."""
    m = scrypt.decrypt(self.ciphertext, self.password, self.five_minutes, self.one_megabyte, 0.0625)
    self.assertEqual(m, self.input)