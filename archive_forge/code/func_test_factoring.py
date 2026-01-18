import os
import pickle
from pickle import PicklingError
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def test_factoring(self):
    rsaObj = self.rsa.construct([self.n, self.e, self.d])
    self.assertTrue(rsaObj.p == self.p or rsaObj.p == self.q)
    self.assertTrue(rsaObj.q == self.p or rsaObj.q == self.q)
    self.assertTrue(rsaObj.q * rsaObj.p == self.n)
    self.assertRaises(ValueError, self.rsa.construct, [self.n, self.e, self.n - 1])