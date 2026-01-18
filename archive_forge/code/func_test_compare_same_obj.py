import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_compare_same_obj(self):
    k1 = self.module.StaticTuple('foo', 'bar')
    self.assertCompareEqual(k1, k1)
    k2 = self.module.StaticTuple(k1, k1)
    self.assertCompareEqual(k2, k2)
    k3 = self.module.StaticTuple('foo', 1, None, 'µ', 1.2, 2 ** 65, True, k1)
    self.assertCompareEqual(k3, k3)