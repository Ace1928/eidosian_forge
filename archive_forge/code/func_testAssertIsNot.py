import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertIsNot(self):
    self.assertMessages('assertIsNot', (None, None), ['^unexpectedly identical: None$', '^oops$', '^unexpectedly identical: None$', '^unexpectedly identical: None : oops$'])