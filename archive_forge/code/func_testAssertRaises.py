import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertRaises(self):
    self.assertMessagesCM('assertRaises', (TypeError,), lambda: None, ['^TypeError not raised$', '^oops$', '^TypeError not raised$', '^TypeError not raised : oops$'])