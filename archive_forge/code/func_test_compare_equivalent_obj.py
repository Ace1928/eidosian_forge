import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_compare_equivalent_obj(self):
    k1 = self.module.StaticTuple('foo', 'bar')
    k2 = self.module.StaticTuple('foo', 'bar')
    self.assertCompareEqual(k1, k2)
    k3 = self.module.StaticTuple(k1, k2)
    k4 = self.module.StaticTuple(k2, k1)
    self.assertCompareEqual(k1, k2)
    k5 = self.module.StaticTuple('foo', 1, None, 'µ', 1.2, 2 ** 65, True, k1)
    k6 = self.module.StaticTuple('foo', 1, None, 'µ', 1.2, 2 ** 65, True, k1)
    self.assertCompareEqual(k5, k6)
    k7 = self.module.StaticTuple(None)
    k8 = self.module.StaticTuple(None)
    self.assertCompareEqual(k7, k8)