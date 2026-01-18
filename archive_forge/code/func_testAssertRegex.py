import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertRegex(self):
    self.assertMessages('assertRegex', ('foo', 'bar'), ["^Regex didn't match:", '^oops$', "^Regex didn't match:", "^Regex didn't match: (.*) : oops$"])