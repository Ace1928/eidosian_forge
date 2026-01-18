import os
import pickle
from pickle import PicklingError
from Cryptodome.Util.py3compat import *
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
def test_construct_2tuple(self):
    """RSA (default implementation) constructed key (2-tuple)"""
    pub = self.rsa.construct((self.n, self.e))
    self._check_public_key(pub)
    self._check_encryption(pub)