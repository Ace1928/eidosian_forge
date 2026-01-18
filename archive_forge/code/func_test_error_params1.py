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
def test_error_params1(self):
    self.assertRaises(ValueError, self.ref_private.export_key, format='XXX')
    self.assertRaises(ValueError, self.ref_private.export_key, format='PEM', passphrase='secret')
    self.assertRaises(ValueError, self.ref_private.export_key, format='PEM', passphrase='', use_pkcs8=False)
    self.assertRaises(ValueError, self.ref_private.export_key, format='PEM', passphrase='', protection='PBKDF2WithHMAC-SHA1AndAES128-CBC')
    self.assertRaises(ValueError, self.ref_private.export_key, format='OpenSSH', passphrase='secret')