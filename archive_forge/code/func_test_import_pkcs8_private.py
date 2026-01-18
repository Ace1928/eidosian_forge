import os
import re
import errno
import warnings
import unittest
from Cryptodome.PublicKey import RSA
from Cryptodome.SelfTest.st_common import a2b_hex, list_test_cases
from Cryptodome.IO import PEM
from Cryptodome.Util.py3compat import b, tostr, FileNotFoundError
from Cryptodome.Util.number import inverse, bytes_to_long
from Cryptodome.Util import asn1
def test_import_pkcs8_private(self):
    key_file_ref = load_file('rsa2048_private.pem')
    key_file = load_file('rsa2048_private_p8.der')
    if None in (key_file_ref, key_file):
        return
    key_ref = RSA.import_key(key_file_ref)
    key = RSA.import_key(key_file, b'secret')
    self.assertEqual(key_ref, key)