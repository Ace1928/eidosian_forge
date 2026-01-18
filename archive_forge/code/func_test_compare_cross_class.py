import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_compare_cross_class(self):
    k1 = self.module.StaticTuple('baz', 'bing')
    self.assertCompareNoRelation(10, k1, mismatched_types=True)
    self.assertCompareNoRelation('baz', k1, mismatched_types=True)