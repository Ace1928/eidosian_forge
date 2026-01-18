import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_compare_all_different_same_width(self):
    k1 = self.module.StaticTuple('baz', 'bing')
    k2 = self.module.StaticTuple('foo', 'bar')
    self.assertCompareDifferent(k1, k2)
    k3 = self.module.StaticTuple(k1, k2)
    k4 = self.module.StaticTuple(k2, k1)
    self.assertCompareDifferent(k3, k4)
    k5 = self.module.StaticTuple(1)
    k6 = self.module.StaticTuple(2)
    self.assertCompareDifferent(k5, k6)
    k7 = self.module.StaticTuple(1.2)
    k8 = self.module.StaticTuple(2.4)
    self.assertCompareDifferent(k7, k8)
    k9 = self.module.StaticTuple('sµ')
    k10 = self.module.StaticTuple('så')
    self.assertCompareDifferent(k9, k10)