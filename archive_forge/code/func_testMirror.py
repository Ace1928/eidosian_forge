import spherogram
import unittest
import doctest
import sys
from random import randrange
from . import test_montesinos
from ...sage_helper import _within_sage
def testMirror(self):
    return
    repeat = 3
    while repeat > 0:
        k1 = self.random_knot()
        k1_prime = k1.mirror()
        self.assertTrue(k1.signature() == -1 * k1_prime.signature(), msg='knot signature failed for ' + repr(k1))
        self.assertTrue(k1.writhe() == -1 * k1_prime.writhe(), msg='knot writhe failed for ' + repr(k1))
        repeat -= 1
    repeat = 3
    while repeat > 0:
        k2 = self.random_link()
        k2_prime = k2.mirror()
        self.assertTrue(k2.signature() == -1 * k2_prime.signature(), msg='link signature failed for ' + repr(k2))
        self.assertTrue(k2.writhe() == -1 * k2_prime.writhe(), msg='link writhe failed for ' + repr(k2))
        repeat -= 1