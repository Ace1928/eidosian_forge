import unittest
from Cryptodome.IO._PBES import PBES2
def test5(self):
    ct = PBES2.encrypt(self.ref, self.passphrase, 'PBKDF2WithHMAC-SHA512AndAES128-GCM')
    pt = PBES2.decrypt(ct, self.passphrase)
    self.assertEqual(self.ref, pt)