import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_ffffffffffff_d0b(self):
    tv = 'BB 36 76 49 51 EC 97 E9 D8 5F 7E E9 A6 7A 77 18\n        FC 00 5C F4 25 56 BE 79 CE 12 C0 BD E5 0E 57 36\n        D6 63 2B 0D 0D FB 20 2D 1B BB 8F FE 3D D7 4C B0\n        08 34 FA 75 6C B0 34 71 BA B1 3A 1E 2C 16 B3 C0'
    btv = txt2bin(tv)
    res = TurboSHAKE256.new(data=b'\xff' * 7, domain=11).read(64)
    self.assertEqual(res, btv)