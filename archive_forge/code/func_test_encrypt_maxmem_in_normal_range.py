import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_encrypt_maxmem_in_normal_range(self):
    """Test encrypt maxmem accepts (> 1 megabyte) of storage to use for V array."""
    s = scrypt.encrypt(self.input, self.password, 0.01, self.ten_megabytes)
    m = scrypt.decrypt(s, self.password)
    self.assertEqual(m, self.input)