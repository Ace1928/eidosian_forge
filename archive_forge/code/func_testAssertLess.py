import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertLess(self):
    self.assertMessages('assertLess', (2, 1), ['^2 not less than 1$', '^oops$', '^2 not less than 1$', '^2 not less than 1 : oops$'])