import gc
import sys
import unittest as pyunit
import weakref
from io import StringIO
from twisted.internet import defer, reactor
from twisted.python.compat import _PYPY
from twisted.python.reflect import namedAny
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import (
from twisted.trial.test import erroneous
from twisted.trial.test.test_suppression import SuppressionMixin
def test_clearPyunitSuite(self):
    """
        Calling L{_clearSuite} on a populated standard library
        L{TestSuite} removes all tests.

        This test is important since C{_clearSuite} operates by mutating
        internal variables.
        """
    suite = pyunit.TestSuite()
    suite.addTest(self.TestCase())
    self.assertEqual(1, suite.countTestCases())
    _clearSuite(suite)
    self.assertEqual(0, suite.countTestCases())