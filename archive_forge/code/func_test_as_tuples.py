import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_as_tuples(self):
    k1 = self.module.StaticTuple('foo', 'bar')
    t = static_tuple.as_tuples(k1)
    self.assertIsInstance(t, tuple)
    self.assertEqual(('foo', 'bar'), t)
    k2 = self.module.StaticTuple(1, k1)
    t = static_tuple.as_tuples(k2)
    self.assertIsInstance(t, tuple)
    self.assertIsInstance(t[1], tuple)
    self.assertEqual((1, ('foo', 'bar')), t)
    mixed = (1, k1)
    t = static_tuple.as_tuples(mixed)
    self.assertIsInstance(t, tuple)
    self.assertIsInstance(t[1], tuple)
    self.assertEqual((1, ('foo', 'bar')), t)