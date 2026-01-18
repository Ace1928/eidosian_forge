import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_ffffff_d07(self):
    tv = '18 B3 B5 B7 06 1C 2E 67 C1 75 3A 00 E6 AD 7E D7\n        BA 1C 90 6C F9 3E FB 70 92 EA F2 7F BE EB B7 55\n        AE 6E 29 24 93 C1 10 E4 8D 26 00 28 49 2B 8E 09\n        B5 50 06 12 B8 F2 57 89 85 DE D5 35 7D 00 EC 67'
    btv = txt2bin(tv)
    res = TurboSHAKE256.new(data=b'\xff' * 3, domain=7).read(64)
    self.assertEqual(res, btv)