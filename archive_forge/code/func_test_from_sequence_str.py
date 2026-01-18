import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_from_sequence_str(self):
    st = self.module.StaticTuple.from_sequence('foo')
    self.assertIsInstance(st, self.module.StaticTuple)
    self.assertEqual(('f', 'o', 'o'), st)