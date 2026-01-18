import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertGreater(self):
    self.assertMessages('assertGreater', (1, 2), ['^1 not greater than 2$', '^oops$', '^1 not greater than 2$', '^1 not greater than 2 : oops$'])