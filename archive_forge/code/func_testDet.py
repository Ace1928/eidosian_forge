import spherogram
import unittest
import doctest
import sys
from random import randrange
from . import test_montesinos
from ...sage_helper import _within_sage
def testDet(self):
    self.assertEqual(self.K3_1.determinant(), 3)
    self.assertEqual(self.K3_1.determinant(method='color'), 3)
    self.assertEqual(self.K3_1.determinant(method='goeritz'), 3)
    self.assertEqual(self.Tref.determinant(), 3)
    self.assertEqual(self.Tref.determinant(method='color'), 3)
    self.assertEqual(self.Tref.determinant(method='goeritz'), 3)
    self.assertEqual(self.K7_2.determinant(), 11)
    self.assertEqual(self.K7_2.determinant(method='color'), 11)
    self.assertEqual(self.K7_2.determinant(method='goeritz'), 11)
    self.assertEqual(self.K8_3.determinant(), 17)
    self.assertEqual(self.K8_3.determinant(method='color'), 17)
    self.assertEqual(self.K8_3.determinant(method='goeritz'), 17)
    self.assertEqual(self.K8_13.determinant(), 29)
    self.assertEqual(self.K8_13.determinant(method='color'), 29)
    self.assertEqual(self.K8_13.determinant(method='goeritz'), 29)
    self.assertEqual(self.L2a1.determinant(), 2)
    self.assertEqual(self.L2a1.determinant(method='color'), 2)
    self.assertEqual(self.L2a1.determinant(method='goeritz'), 2)
    self.assertEqual(self.L6a2.determinant(), 10)
    self.assertEqual(self.L6a2.determinant(method='color'), 10)
    self.assertEqual(self.L6a2.determinant(method='goeritz'), 10)
    self.assertEqual(self.Borr.determinant(), 16)
    self.assertEqual(self.Borr.determinant(method='color'), 16)
    self.assertEqual(self.Borr.determinant(method='goeritz'), 16)
    self.assertEqual(self.L6a4.determinant(), 16)
    self.assertEqual(self.L6a4.determinant(method='color'), 16)
    self.assertEqual(self.L6a4.determinant(method='goeritz'), 16)
    self.assertEqual(self.L7a3.determinant(), 16)
    self.assertEqual(self.L7a3.determinant(method='color'), 16)
    self.assertEqual(self.L7a3.determinant(method='goeritz'), 16)