import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_compare_diff_width(self):
    k1 = self.module.StaticTuple('foo')
    k2 = self.module.StaticTuple('foo', 'bar')
    self.assertCompareDifferent(k1, k2)
    k3 = self.module.StaticTuple(k1)
    k4 = self.module.StaticTuple(k1, k2)
    self.assertCompareDifferent(k3, k4)