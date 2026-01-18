import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_holds_int(self):
    k1 = self.module.StaticTuple(1)

    class subint(int):
        pass
    self.assertRaises(TypeError, self.module.StaticTuple, subint(2))