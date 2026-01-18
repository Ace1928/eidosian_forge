import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testNotAlmostEqual(self):
    self.assertMessages('assertNotAlmostEqual', (1, 1), ['^1 == 1 within 7 places$', '^oops$', '^1 == 1 within 7 places$', '^1 == 1 within 7 places : oops$'])