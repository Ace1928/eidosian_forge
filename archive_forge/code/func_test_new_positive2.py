import unittest
from binascii import hexlify, unhexlify
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import keccak
from Cryptodome.Util.py3compat import b, tobytes, bchr
def test_new_positive2(self):
    digest1 = keccak.new(data=b('\x90'), digest_bytes=64).digest()
    digest2 = keccak.new(digest_bytes=64).update(b('\x90')).digest()
    self.assertEqual(digest1, digest2)