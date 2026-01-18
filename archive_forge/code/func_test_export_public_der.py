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
def test_export_public_der(self):
    key_file = load_file('ecc_ed448_public.der')
    encoded = self.ref_public._export_subjectPublicKeyInfo(True)
    self.assertEqual(key_file, encoded)
    encoded = self.ref_public.export_key(format='DER')
    self.assertEqual(key_file, encoded)
    encoded = self.ref_public.export_key(format='DER', compress=False)
    self.assertEqual(key_file, encoded)