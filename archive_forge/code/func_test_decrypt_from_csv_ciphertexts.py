import unittest as testm
from binascii import a2b_hex, b2a_hex
from csv import reader
from os import urandom
from os.path import abspath, dirname, sep
import scrypt
def test_decrypt_from_csv_ciphertexts(self):
    """Test decrypt function with precalculated combinations."""
    for row in self.ciphertexts[1:]:
        h = scrypt.decrypt(a2b_hex(bytes(row[5].encode('ascii'))), row[1])
        self.assertEqual(bytes(h.encode('ascii')), row[0].encode('ascii'))