import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_decrypt_maxmemfrac_keyword_argument(self):
    """Test decrypt function accepts maxmem keyword argument."""
    m = scrypt.decrypt(maxmemfrac=0.625, input=self.ciphertext, password=self.password)
    self.assertEqual(m, self.input)