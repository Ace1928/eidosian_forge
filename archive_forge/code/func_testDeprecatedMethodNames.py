import contextlib
import difflib
import pprint
import pickle
import re
import sys
import logging
import warnings
import weakref
import inspect
import types
from copy import deepcopy
from test import support
import unittest
from unittest.test.support import (
from test.support import captured_stderr, gc_collect
def testDeprecatedMethodNames(self):
    """
        Test that the deprecated methods raise a DeprecationWarning. See #9424.
        """
    old = ((self.failIfEqual, (3, 5)), (self.assertNotEquals, (3, 5)), (self.failUnlessEqual, (3, 3)), (self.assertEquals, (3, 3)), (self.failUnlessAlmostEqual, (2.0, 2.0)), (self.assertAlmostEquals, (2.0, 2.0)), (self.failIfAlmostEqual, (3.0, 5.0)), (self.assertNotAlmostEquals, (3.0, 5.0)), (self.failUnless, (True,)), (self.assert_, (True,)), (self.failUnlessRaises, (TypeError, lambda _: 3.14 + 'spam')), (self.failIf, (False,)), (self.assertDictContainsSubset, (dict(a=1, b=2), dict(a=1, b=2, c=3))), (self.assertRaisesRegexp, (KeyError, 'foo', lambda: {}['foo'])), (self.assertRegexpMatches, ('bar', 'bar')))
    for meth, args in old:
        with self.assertWarns(DeprecationWarning):
            meth(*args)