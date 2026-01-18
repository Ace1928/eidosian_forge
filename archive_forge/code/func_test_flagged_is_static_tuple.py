import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_flagged_is_static_tuple(self):
    debug.debug_flags.add('static_tuple')
    st = static_tuple.StaticTuple('foo')
    st2 = static_tuple.expect_static_tuple(st)
    self.assertIs(st, st2)