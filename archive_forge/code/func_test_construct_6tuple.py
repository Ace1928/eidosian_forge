import os
import pickle
from pickle import PicklingError
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def test_construct_6tuple(self):
    """RSA (default implementation) constructed key (6-tuple)"""
    rsaObj = self.rsa.construct((self.n, self.e, self.d, self.p, self.q, self.u))
    self._check_private_key(rsaObj)
    self._check_encryption(rsaObj)
    self._check_decryption(rsaObj)