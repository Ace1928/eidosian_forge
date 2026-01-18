import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertFalse(self):
    self.assertMessages('assertFalse', (True,), ['^True is not false$', '^oops$', '^True is not false$', '^True is not false : oops$'])