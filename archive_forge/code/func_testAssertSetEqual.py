import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertSetEqual(self):
    self.assertMessages('assertSetEqual', (set(), set([None])), ['None$', '^oops$', 'None$', 'None : oops$'])