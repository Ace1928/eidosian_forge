import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import tobytes, bord, bchr
from Cryptodome.Hash import (SHA1, SHA224, SHA256, SHA384, SHA512,
from Cryptodome.Signature import DSS
from Cryptodome.PublicKey import DSA, ECC
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
def test_negative_unapproved_hashes(self):
    """Verify that unapproved hashes are rejected"""
    from Cryptodome.Hash import SHA1
    self.description = 'Unapproved hash (SHA-1) test'
    hash_obj = SHA1.new()
    signer = DSS.new(self.key_priv, 'fips-186-3')
    self.assertRaises(ValueError, signer.sign, hash_obj)
    self.assertRaises(ValueError, signer.verify, hash_obj, b'\x00' * 40)