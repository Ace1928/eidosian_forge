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
def test_negative_unknown_modes_encodings(self):
    """Verify that unknown modes/encodings are rejected"""
    self.description = 'Unknown mode test'
    self.assertRaises(ValueError, DSS.new, self.key_priv, 'fips-186-0')
    self.description = 'Unknown encoding test'
    self.assertRaises(ValueError, DSS.new, self.key_priv, 'fips-186-3', 'xml')