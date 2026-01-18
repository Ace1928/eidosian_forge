import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_encrypt_maxtime_key(self):
    """Test encrypt maxtime accepts maxtime as keyword argument."""
    s = scrypt.encrypt(self.input, self.password, maxtime=0.01)
    m = scrypt.decrypt(s, self.password)
    self.assertEqual(m, self.input)