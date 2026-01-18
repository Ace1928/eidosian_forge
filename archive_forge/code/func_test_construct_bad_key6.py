import os
import pickle
from pickle import PicklingError
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def test_construct_bad_key6(self):
    tup = (self.n, self.e, self.d, self.p, self.q, 10)
    self.assertRaises(ValueError, self.rsa.construct, tup)
    from Cryptodome.Util.number import inverse
    tup = (self.n, self.e, self.d, self.p, self.q, inverse(self.q, self.p))
    self.assertRaises(ValueError, self.rsa.construct, tup)