import os
import errno
import warnings
import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import bord, tostr, FileNotFoundError
from Cryptodome.Util.asn1 import DerSequence, DerBitString
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Hash import SHAKE128
from Cryptodome.PublicKey import ECC
def test_import_private_pem_encrypted(self):
    for algo in ('des3', 'aes128', 'aes192', 'aes256'):
        key_file = load_file('ecc_ed448_private_enc_%s.pem' % algo)
        key = ECC.import_key(key_file, 'secret')
        self.assertEqual(self.ref_private, key)
        key = ECC.import_key(tostr(key_file), b'secret')
        self.assertEqual(self.ref_private, key)