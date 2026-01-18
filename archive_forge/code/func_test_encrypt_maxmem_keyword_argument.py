import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_encrypt_maxmem_keyword_argument(self):
    """Test encrypt maxmem accepts exactly (1 megabyte) of storage to use for V
        array."""
    s = scrypt.encrypt(self.input, self.password, maxmem=self.one_megabyte, maxtime=0.01)
    m = scrypt.decrypt(s, self.password)
    self.assertEqual(m, self.input)