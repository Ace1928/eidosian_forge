import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_ffffff_d01(self):
    tv = 'D2 1C 6F BB F5 87 FA 22 82 F2 9A EA 62 01 75 FB\n        02 57 41 3A F7 8A 0B 1B 2A 87 41 9C E0 31 D9 33\n        AE 7A 4D 38 33 27 A8 A1 76 41 A3 4F 8A 1D 10 03\n        AD 7D A6 B7 2D BA 84 BB 62 FE F2 8F 62 F1 24 24'
    btv = txt2bin(tv)
    res = TurboSHAKE256.new(data=b'\xff\xff\xff', domain=1).read(64)
    self.assertEqual(res, btv)