import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
from Cryptodome import Random
from Cryptodome.PublicKey import ElGamal
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Util.py3compat import *
def test_decryption(self):
    for tv in self.tve:
        d = self.convert_tv(tv, True)
        key = ElGamal.construct(d['key'])
        pt = key._decrypt((d['ct1'], d['ct2']))
        self.assertEqual(pt, d['pt'])