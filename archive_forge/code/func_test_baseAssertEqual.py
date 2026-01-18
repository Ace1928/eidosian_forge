import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def test_baseAssertEqual(self):
    self.assertMessages('_baseAssertEqual', (1, 2), ['^1 != 2$', '^oops$', '^1 != 2$', '^1 != 2 : oops$'])