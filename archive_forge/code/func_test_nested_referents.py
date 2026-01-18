import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_nested_referents(self):
    self.requireFeature(features.meliae)
    from meliae import scanner
    strs = ['foo', 'bar', 'baz', 'bing']
    k1 = self.module.StaticTuple(*strs[:2])
    k2 = self.module.StaticTuple(*strs[2:])
    k3 = self.module.StaticTuple(k1, k2)
    refs = [k1, k2]
    if self.module is _static_tuple_py:
        refs.append(self.module.StaticTuple)

    def key(k):
        if isinstance(k, type):
            return (0, k)
        if isinstance(k, self.module.StaticTuple):
            return (1, k)
        raise TypeError(k)
    self.assertEqual(sorted(refs, key=key), sorted(scanner.get_referents(k3), key=key))