import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertDictEqual(self):
    self.assertMessages('assertDictEqual', ({}, {'key': 'value'}), ["\\+ \\{'key': 'value'\\}$", '^oops$', "\\+ \\{'key': 'value'\\}$", "\\+ \\{'key': 'value'\\} : oops$"])