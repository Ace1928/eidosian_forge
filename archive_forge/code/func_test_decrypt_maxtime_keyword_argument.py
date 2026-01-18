import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_decrypt_maxtime_keyword_argument(self):
    """Test decrypt function accepts maxtime keyword argument."""
    m = scrypt.decrypt(maxtime=1.0, input=self.ciphertext, password=self.password)
    self.assertEqual(m, self.input)