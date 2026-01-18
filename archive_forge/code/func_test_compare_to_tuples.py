import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_compare_to_tuples(self):
    k1 = self.module.StaticTuple('foo')
    self.assertCompareEqual(k1, ('foo',))
    self.assertCompareEqual(('foo',), k1)
    self.assertCompareDifferent(k1, ('foo', 'bar'))
    self.assertCompareDifferent(k1, ('foo', 10))
    k2 = self.module.StaticTuple('foo', 'bar')
    self.assertCompareEqual(k2, ('foo', 'bar'))
    self.assertCompareEqual(('foo', 'bar'), k2)
    self.assertCompareDifferent(k2, ('foo', 'zzz'))
    self.assertCompareDifferent(('foo',), k2)
    self.assertCompareDifferent(('foo', 'aaa'), k2)
    self.assertCompareDifferent(('baz', 'bing'), k2)
    self.assertCompareDifferent(('foo', 10), k2, mismatched_types=True)
    k3 = self.module.StaticTuple(k1, k2)
    self.assertCompareEqual(k3, (('foo',), ('foo', 'bar')))
    self.assertCompareEqual((('foo',), ('foo', 'bar')), k3)
    self.assertCompareEqual(k3, (k1, ('foo', 'bar')))
    self.assertCompareEqual((k1, ('foo', 'bar')), k3)