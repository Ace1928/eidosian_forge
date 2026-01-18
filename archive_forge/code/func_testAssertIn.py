import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertIn(self):
    self.assertMessages('assertIn', (None, []), ['^None not found in \\[\\]$', '^oops$', '^None not found in \\[\\]$', '^None not found in \\[\\] : oops$'])