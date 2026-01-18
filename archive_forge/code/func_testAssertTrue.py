import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertTrue(self):
    self.assertMessages('assertTrue', (False,), ['^False is not true$', '^oops$', '^False is not true$', '^False is not true : oops$'])