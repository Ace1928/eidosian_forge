import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertSequenceEqual(self):
    self.assertMessages('assertSequenceEqual', ([], [None]), ['\\+ \\[None\\]$', '^oops$', '\\+ \\[None\\]$', '\\+ \\[None\\] : oops$'])