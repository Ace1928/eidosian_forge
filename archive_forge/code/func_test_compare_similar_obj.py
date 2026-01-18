import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_compare_similar_obj(self):
    k1 = self.module.StaticTuple('foo' + ' bar', 'bar' + ' baz')
    k2 = self.module.StaticTuple('fo' + 'o bar', 'ba' + 'r baz')
    self.assertCompareEqual(k1, k2)
    k3 = self.module.StaticTuple('foo ' + 'bar', 'bar ' + 'baz')
    k4 = self.module.StaticTuple('f' + 'oo bar', 'b' + 'ar baz')
    k5 = self.module.StaticTuple(k1, k2)
    k6 = self.module.StaticTuple(k3, k4)
    self.assertCompareEqual(k5, k6)