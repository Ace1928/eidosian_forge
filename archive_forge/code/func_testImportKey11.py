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
def testImportKey11(self):
    """Verify import of RSAPublicKey DER SEQUENCE"""
    der = asn1.DerSequence([17, 3]).encode()
    key = RSA.importKey(der)
    self.assertEqual(key.n, 17)
    self.assertEqual(key.e, 3)