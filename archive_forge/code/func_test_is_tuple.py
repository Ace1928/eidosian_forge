import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_is_tuple(self):
    t = ('foo',)
    st = static_tuple.expect_static_tuple(t)
    self.assertIsInstance(st, static_tuple.StaticTuple)
    self.assertEqual(t, st)