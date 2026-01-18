import unittest
from Cryptodome.IO._PBES import PBES2
def test7(self):
    ct = PBES2.encrypt(self.ref, self.passphrase, 'PBKDF2WithHMAC-SHA3-256AndAES256-GCM')
    pt = PBES2.decrypt(ct, self.passphrase)
    self.assertEqual(self.ref, pt)