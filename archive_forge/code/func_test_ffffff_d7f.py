import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_ffffff_d7f(self):
    tv = 'AB E5 69 C1 F7 7E C3 40 F0 27 05 E7 D3 7C 9A B7\n        E1 55 51 6E 4A 6A 15 00 21 D7 0B 6F AC 0B B4 0C\n        06 9F 9A 98 28 A0 D5 75 CD 99 F9 BA E4 35 AB 1A\n        CF 7E D9 11 0B A9 7C E0 38 8D 07 4B AC 76 87 76'
    btv = txt2bin(tv)
    res = TurboSHAKE256.new(data=b'\xff' * 3, domain=127).read(64)
    self.assertEqual(res, btv)