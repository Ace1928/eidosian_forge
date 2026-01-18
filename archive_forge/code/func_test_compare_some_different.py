import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_compare_some_different(self):
    k1 = self.module.StaticTuple('foo', 'bar')
    k2 = self.module.StaticTuple('foo', 'zzz')
    self.assertCompareDifferent(k1, k2)
    k3 = self.module.StaticTuple(k1, k1)
    k4 = self.module.StaticTuple(k1, k2)
    self.assertCompareDifferent(k3, k4)
    k5 = self.module.StaticTuple('foo', None)
    self.assertCompareDifferent(k5, k1, mismatched_types=True)
    self.assertCompareDifferent(k5, k2, mismatched_types=True)