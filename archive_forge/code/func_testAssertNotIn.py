import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertNotIn(self):
    self.assertMessages('assertNotIn', (None, [None]), ['^None unexpectedly found in \\[None\\]$', '^oops$', '^None unexpectedly found in \\[None\\]$', '^None unexpectedly found in \\[None\\] : oops$'])