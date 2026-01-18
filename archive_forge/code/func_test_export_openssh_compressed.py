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
def test_export_openssh_compressed(self):
    key_file = load_file('ecc_p521_public_openssh.txt', 'rt')
    pub_key = ECC.import_key(key_file)
    key_file_compressed = pub_key.export_key(format='OpenSSH', compress=True)
    assert len(key_file) > len(key_file_compressed)
    self.assertEqual(pub_key, ECC.import_key(key_file_compressed))