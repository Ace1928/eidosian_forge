import spherogram
import unittest
import doctest
import sys
from random import randrange
from . import test_montesinos
from ...sage_helper import _within_sage
def testKnotGroup(self):
    for k in self.all_links:
        self.assertEqual(len(k.knot_group().generators()), len(k.crossings))
        self.assertEqual(k.knot_group().abelian_invariants()[0], 0)
        self.assertEqual(len(k.knot_group().abelian_invariants()), len(k.link_components))