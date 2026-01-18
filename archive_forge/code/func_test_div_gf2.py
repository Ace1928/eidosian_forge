from unittest import main, TestCase, TestSuite
from binascii import unhexlify, hexlify
from Cryptodome.Util.py3compat import *
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Protocol.SecretSharing import Shamir, _Element, \
def test_div_gf2(self):
    from Cryptodome.Util.number import size as deg
    x, y = _div_gf2(567, 7)
    self.assertTrue(deg(y) < deg(7))
    w = _mult_gf2(x, 7) ^ y
    self.assertEqual(567, w)
    x, y = _div_gf2(7, 567)
    self.assertEqual(x, 0)
    self.assertEqual(y, 7)