import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_create_bad_args(self):
    args_256 = ['a'] * 256
    self.assertRaises(TypeError, self.module.StaticTuple, *args_256)
    args_300 = ['a'] * 300
    self.assertRaises(TypeError, self.module.StaticTuple, *args_300)
    self.assertRaises(TypeError, self.module.StaticTuple, object())