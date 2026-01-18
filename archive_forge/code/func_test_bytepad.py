import unittest
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import cSHAKE128, cSHAKE256, SHAKE128, SHAKE256
from Cryptodome.Util.py3compat import b, bchr, tobytes
def test_bytepad(self):
    from Cryptodome.Hash.cSHAKE128 import _bytepad
    self.assertEqual(_bytepad(b'', 4), b'\x01\x04\x00\x00')
    self.assertEqual(_bytepad(b'A', 4), b'\x01\x04A\x00')
    self.assertEqual(_bytepad(b'AA', 4), b'\x01\x04AA')
    self.assertEqual(_bytepad(b'AAA', 4), b'\x01\x04AAA\x00\x00\x00')
    self.assertEqual(_bytepad(b'AAAA', 4), b'\x01\x04AAAA\x00\x00')
    self.assertEqual(_bytepad(b'AAAAA', 4), b'\x01\x04AAAAA\x00')
    self.assertEqual(_bytepad(b'AAAAAA', 4), b'\x01\x04AAAAAA')
    self.assertEqual(_bytepad(b'AAAAAAA', 4), b'\x01\x04AAAAAAA\x00\x00\x00')