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
def test_import_sec1_uncompressed(self):
    key_file = load_file('ecc_p521_public.der')
    value = extract_bitstring_from_spki(key_file)
    key = ECC.import_key(key_file, curve_name='P521')
    self.assertEqual(self.ref_public, key)