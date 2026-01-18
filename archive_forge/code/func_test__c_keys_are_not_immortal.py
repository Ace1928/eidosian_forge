import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test__c_keys_are_not_immortal(self):
    if self.module is _static_tuple_py:
        return
    unique_str1 = 'unique str ' + osutils.rand_chars(20)
    unique_str2 = 'unique str ' + osutils.rand_chars(20)
    key = self.module.StaticTuple(unique_str1, unique_str2)
    self.assertFalse(key in self.module._interned_tuples)
    self.assertRefcount(1, key)
    key = key.intern()
    self.assertRefcount(1, key)
    self.assertTrue(key in self.module._interned_tuples)
    self.assertTrue(key._is_interned())
    del key
    key = self.module.StaticTuple(unique_str1, unique_str2)
    self.assertRefcount(1, key)
    self.assertFalse(key in self.module._interned_tuples)
    self.assertFalse(key._is_interned())