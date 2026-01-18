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
def test_export_private_pkcs8_encrypted(self):
    encoded = self.ref_private._export_pkcs8(passphrase='secret', protection='PBKDF2WithHMAC-SHA1AndAES128-CBC')
    self.assertRaises(ValueError, ECC._import_pkcs8, encoded, None)
    decoded = ECC._import_pkcs8(encoded, 'secret')
    self.assertEqual(self.ref_private, decoded)
    encoded = self.ref_private.export_key(format='DER', passphrase='secret', protection='PBKDF2WithHMAC-SHA1AndAES128-CBC')
    decoded = ECC.import_key(encoded, 'secret')
    self.assertEqual(self.ref_private, decoded)
    encoded = self.ref_private.export_key(format='DER', passphrase='secret', protection='PBKDF2WithHMAC-SHA384AndAES128-CBC', prot_params={'iteration_count': 123})
    decoded = ECC.import_key(encoded, 'secret')
    self.assertEqual(self.ref_private, decoded)