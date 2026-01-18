import spherogram
import unittest
import doctest
import sys
from random import randrange
from . import test_montesinos
from ...sage_helper import _within_sage
def testConnectedSum(self):
    repeat = 3
    while repeat > 0:
        k1 = self.random_knot()
        k2 = self.random_knot()
        Sum = k1.connected_sum(k2)
        self.assertEqual(Sum.alexander_polynomial(), k1.alexander_polynomial() * k2.alexander_polynomial())
        self.assertEqual(Sum.signature(), k1.signature() + k2.signature())
        self.assertEqual(Sum.determinant(), k1.determinant() * k2.determinant())
        repeat -= 1