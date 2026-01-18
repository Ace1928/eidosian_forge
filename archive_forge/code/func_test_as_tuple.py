import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_as_tuple(self):
    k = self.module.StaticTuple('foo')
    t = k.as_tuple()
    self.assertEqual(('foo',), t)
    self.assertIsInstance(t, tuple)
    self.assertFalse(isinstance(t, self.module.StaticTuple))
    k = self.module.StaticTuple('foo', 'bar')
    t = k.as_tuple()
    self.assertEqual(('foo', 'bar'), t)
    k2 = self.module.StaticTuple(1, k)
    t = k2.as_tuple()
    self.assertIsInstance(t, tuple)
    self.assertIsInstance(t[1], self.module.StaticTuple)
    self.assertEqual((1, ('foo', 'bar')), t)