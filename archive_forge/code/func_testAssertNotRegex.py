import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertNotRegex(self):
    self.assertMessages('assertNotRegex', ('foo', 'foo'), ['^Regex matched:', '^oops$', '^Regex matched:', '^Regex matched: (.*) : oops$'])