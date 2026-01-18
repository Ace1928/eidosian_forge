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
def test_byte_or_string_passphrase(self):
    encoded1 = self.ref_private.export_key(format='PEM', passphrase='secret', protection='PBKDF2WithHMAC-SHA1AndAES128-CBC', randfunc=get_fixed_prng())
    encoded2 = self.ref_private.export_key(format='PEM', passphrase=b'secret', protection='PBKDF2WithHMAC-SHA1AndAES128-CBC', randfunc=get_fixed_prng())
    self.assertEqual(encoded1, encoded2)