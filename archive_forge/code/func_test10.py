import unittest
from Cryptodome.IO._PBES import PBES2
def test10(self):
    ct = PBES2.encrypt(self.ref, self.passphrase, 'scryptAndAES256-CBC')
    pt = PBES2.decrypt(ct, self.passphrase)
    self.assertEqual(self.ref, pt)