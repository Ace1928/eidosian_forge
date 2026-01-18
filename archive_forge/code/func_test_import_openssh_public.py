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
def test_import_openssh_public(self):
    key_file = load_file('ecc_ed25519_public_openssh.txt')
    key = ECC._import_openssh_public(key_file)
    self.assertFalse(key.has_private())
    key = ECC.import_key(key_file)
    self.assertFalse(key.has_private())