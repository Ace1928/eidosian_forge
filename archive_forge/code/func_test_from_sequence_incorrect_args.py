import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_from_sequence_incorrect_args(self):
    self.assertRaises(TypeError, self.module.StaticTuple.from_sequence, object(), 'a')
    self.assertRaises(TypeError, self.module.StaticTuple.from_sequence, foo='a')