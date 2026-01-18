import spherogram
import unittest
import doctest
import sys
from random import randrange
from . import test_montesinos
from ...sage_helper import _within_sage
def testSignature(self):
    self.assertEqual(abs(self.Tref.signature()), 2)
    self.assertEqual(abs(self.K3_1.signature()), 2)
    self.assertEqual(abs(self.K7_2.signature()), 2)
    self.assertEqual(abs(self.K8_3.signature()), 0)
    self.assertEqual(abs(self.K8_13.signature()), 0)
    self.assertEqual(abs(self.L2a1.signature()), 1)
    self.assertEqual(abs(self.L6a2.signature()), 3)
    self.assertEqual(abs(self.Borr.signature()), 0)
    self.assertEqual(abs(self.L6a4.signature()), 0)
    self.assertEqual(abs(self.L7a3.signature()), 3)