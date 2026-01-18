import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test__c_has_C_API(self):
    if self.module is _static_tuple_py:
        return
    self.assertIsNot(None, self.module._C_API)