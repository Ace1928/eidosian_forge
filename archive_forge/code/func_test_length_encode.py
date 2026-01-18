import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import KangarooTwelve as K12
from Cryptodome.Util.py3compat import b, bchr
def test_length_encode(self):
    self.assertEqual(K12._length_encode(0), b'\x00')
    self.assertEqual(K12._length_encode(12), b'\x0c\x01')
    self.assertEqual(K12._length_encode(65538), b'\x01\x00\x02\x03')