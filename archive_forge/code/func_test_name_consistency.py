import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors
from Cryptodome.PublicKey import ECC
from Cryptodome.PublicKey.ECC import EccPoint, _curves, EccKey
from Cryptodome.Math.Numbers import Integer
def test_name_consistency(self):
    key = ECC.generate(curve='p521')
    self.assertIn("curve='NIST P-521'", repr(key))
    self.assertEqual(key.curve, 'NIST P-521')
    self.assertEqual(key.public_key().curve, 'NIST P-521')