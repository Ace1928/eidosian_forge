import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def assertCompareDifferent(self, k_small, k_big, mismatched_types=False):
    self.assertFalse(k_small == k_big)
    self.assertTrue(k_small != k_big)
    if not self.check_strict_compare(k_small, k_big, mismatched_types):
        self.assertFalse(k_small >= k_big)
        self.assertFalse(k_small > k_big)
        self.assertTrue(k_small <= k_big)
        self.assertTrue(k_small < k_big)