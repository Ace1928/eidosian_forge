import re
import unittest
from binascii import hexlify
from Cryptodome.Util.py3compat import bord
from Cryptodome.Hash import SHA256
from Cryptodome.PublicKey import ECC
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Protocol.DH import key_agreement
def ecdh_test_rev(self, public_key=public_key, private_key=private_key, exp_response=exp_response):
    z = key_agreement(static_pub=public_key, static_priv=private_key, kdf=lambda x: x)
    self.assertEqual(z, exp_response)