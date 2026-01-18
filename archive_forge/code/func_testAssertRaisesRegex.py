import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertRaisesRegex(self):
    self.assertMessagesCM('assertRaisesRegex', (TypeError, 'unused regex'), lambda: None, ['^TypeError not raised$', '^oops$', '^TypeError not raised$', '^TypeError not raised : oops$'])

    def raise_wrong_message():
        raise TypeError('foo')
    self.assertMessagesCM('assertRaisesRegex', (TypeError, 'regex'), raise_wrong_message, ['^"regex" does not match "foo"$', '^oops$', '^"regex" does not match "foo"$', '^"regex" does not match "foo" : oops$'])