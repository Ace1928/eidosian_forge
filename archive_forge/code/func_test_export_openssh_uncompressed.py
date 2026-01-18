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
def test_export_openssh_uncompressed(self):
    key_file = load_file('ecc_p521_public_openssh.txt', 'rt')
    encoded = self.ref_public._export_openssh(False)
    self.assertEqual(key_file, encoded)
    encoded = self.ref_public.export_key(format='OpenSSH')
    self.assertEqual(key_file, encoded)
    encoded = self.ref_public.export_key(format='OpenSSH', compress=False)
    self.assertEqual(key_file, encoded)