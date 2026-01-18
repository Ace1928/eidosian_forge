import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Hash import TurboSHAKE128, TurboSHAKE256
from Cryptodome.Util.py3compat import bchr
def test_ff_d30(self):
    tv = 'F3 FE 12 87 3D 34 BC BB 2E 60 87 79 D6 B7 0E 7F\n        86 BE C7 E9 0B F1 13 CB D4 FD D0 C4 E2 F4 62 5E\n        14 8D D7 EE 1A 52 77 6C F7 7F 24 05 14 D9 CC FC\n        3B 5D DA B8 EE 25 5E 39 EE 38 90 72 96 2C 11 1A'
    btv = txt2bin(tv)
    res = TurboSHAKE256.new(data=b'\xff', domain=48).read(64)
    self.assertEqual(res, btv)