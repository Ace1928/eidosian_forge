import unittest
from Cryptodome.IO._PBES import PBES2
def test8(self):
    ct = PBES2.encrypt(self.ref, self.passphrase, 'scryptAndAES128-CBC')
    pt = PBES2.decrypt(ct, self.passphrase)
    self.assertEqual(self.ref, pt)