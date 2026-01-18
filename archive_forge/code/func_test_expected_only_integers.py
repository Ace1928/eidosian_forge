import unittest
from Cryptodome.Util.py3compat import *
from Cryptodome.Util.asn1 import (DerObject, DerSetOf, DerInteger,
def test_expected_only_integers(self):
    der_bin1 = DerSequence([1, 2, 3]).encode()
    der_bin2 = DerSequence([1, 2, DerSequence([3, 4])]).encode()
    DerSequence().decode(der_bin1, only_ints_expected=True)
    DerSequence().decode(der_bin1, only_ints_expected=False)
    DerSequence().decode(der_bin2, only_ints_expected=False)
    self.assertRaises(ValueError, DerSequence().decode, der_bin2, only_ints_expected=True)