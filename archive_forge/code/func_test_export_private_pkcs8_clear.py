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
def test_export_private_pkcs8_clear(self):
    key_file = load_file('ecc_ed448_private.der')
    encoded = self.ref_private._export_pkcs8()
    self.assertEqual(key_file, encoded)
    encoded = self.ref_private.export_key(format='DER')
    self.assertEqual(key_file, encoded)
    self.assertRaises(ValueError, self.ref_private.export_key, format='DER', use_pkcs8=False)