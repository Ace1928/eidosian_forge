import spherogram
import unittest
import doctest
import sys
from random import randrange
from . import test_montesinos
from ...sage_helper import _within_sage
def testAlexanderPoly(self):
    from sage.all import SR
    L = LaurentPolynomialRing(QQ, 't')
    t = L.gen()
    a = SR.var('a')
    L3v = LaurentPolynomialRing(QQ, ['t1', 't2', 't3'])
    t1, t2, t3 = L3v.gens()
    self.assertEqual(self.Tref.alexander_polynomial(), 1 - t + t ** 2)
    self.assertEqual(self.K3_1.alexander_polynomial(), 1 - t + t ** 2)
    self.assertEqual(self.K7_2.alexander_polynomial(), 3 - 5 * t + 3 * t ** 2)
    self.assertEqual(self.K8_3.alexander_polynomial(), 4 - 9 * t + 4 * t ** 2)
    self.assertEqual(self.K8_13.alexander_polynomial(), 2 - 7 * t + 11 * t ** 2 - 7 * t ** 3 + 2 * t ** 4)
    self.assertEqual(self.L2a1.alexander_polynomial(), 1)
    self.assertEqual(self.Borr.alexander_polynomial(), t1 * t2 * t3 - t1 * t2 - t1 * t3 - t2 * t3 + t1 + t2 + t3 - 1)
    self.assertEqual(self.L6a4.alexander_polynomial(), t1 * t2 * t3 - t1 * t2 - t1 * t3 - t2 * t3 + t1 + t2 + t3 - 1)
    try:
        import snappy
        self.assertEqual(self.Tref.alexander_polynomial(method='snappy'), a ** 2 - a + 1)
        self.assertEqual(self.K3_1.alexander_polynomial(method='snappy'), a ** 2 - a + 1)
        self.assertEqual(self.K7_2.alexander_polynomial(method='snappy'), 3 * a ** 2 - 5 * a + 3)
        self.assertEqual(self.K8_3.alexander_polynomial(method='snappy'), 4 * a ** 2 - 9 * a + 4)
        self.assertEqual(self.K8_13.alexander_polynomial(method='snappy'), 2 * a ** 4 - 7 * a ** 3 + 11 * a ** 2 - 7 * a + 2)
    except ImportError:
        pass