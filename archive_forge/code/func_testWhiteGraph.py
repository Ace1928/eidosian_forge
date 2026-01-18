import spherogram
import unittest
import doctest
import sys
from random import randrange
from . import test_montesinos
from ...sage_helper import _within_sage
def testWhiteGraph(self):
    repeat = 3
    while repeat > 0:
        k1 = self.random_knot()
        self.assertTrue(k1.white_graph().is_planar())
        repeat -= 1
    repeat = 3
    while repeat > 0:
        k2 = self.random_link()
        self.assertTrue(k2.white_graph().is_planar())
        repeat -= 1