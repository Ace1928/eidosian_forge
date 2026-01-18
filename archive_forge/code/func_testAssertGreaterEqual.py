import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertGreaterEqual(self):
    self.assertMessages('assertGreaterEqual', (1, 2), ['^1 not greater than or equal to 2$', '^oops$', '^1 not greater than or equal to 2$', '^1 not greater than or equal to 2 : oops$'])