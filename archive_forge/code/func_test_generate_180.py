import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
from Cryptodome import Random
from Cryptodome.PublicKey import ElGamal
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Util.py3compat import *
def test_generate_180(self):
    self._test_random_key(180)