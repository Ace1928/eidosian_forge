import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_ff_d06(self):
    tv = '73 8D 7B 4E 37 D1 8B 7F 22 AD 1B 53 13 E3 57 E3\n        DD 7D 07 05 6A 26 A3 03 C4 33 FA 35 33 45 52 80\n        F4 F5 A7 D4 F7 00 EF B4 37 FE 6D 28 14 05 E0 7B\n        E3 2A 0A 97 2E 22 E6 3A DC 1B 09 0D AE FE 00 4B'
    btv = txt2bin(tv)
    res = TurboSHAKE256.new(data=b'\xff', domain=6).read(64)
    self.assertEqual(res, btv)