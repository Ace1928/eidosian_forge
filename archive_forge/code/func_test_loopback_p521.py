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
def test_loopback_p521(self):
    hashed_msg = SHA512.new(b'test')
    signer = DSS.new(self.key_priv_p521, 'deterministic-rfc6979')
    signature = signer.sign(hashed_msg)
    verifier = DSS.new(self.key_pub_p521, 'deterministic-rfc6979')
    verifier.verify(hashed_msg, signature)