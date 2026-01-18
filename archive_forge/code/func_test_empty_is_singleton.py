import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_empty_is_singleton(self):
    key = self.module.StaticTuple()
    self.assertIs(key, self.module._empty_tuple)