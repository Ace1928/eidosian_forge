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
def test_expectedSetUpFailure(self):
    """
        C{setUp} is excluded from the failure expectation defined by a C{todo}
        attribute on a test method.
        """
    self.loadSuite(self.SetUpTodo)
    self.suite(self.reporter)
    self.assertFalse(self.reporter.wasSuccessful())
    self.assertEqual(len(self.reporter.errors), 1)
    self.assertEqual(self.reporter.failures, [])
    self.assertEqual(self.reporter.skips, [])
    self.assertEqual(len(self.reporter.expectedFailures), 0)
    self.assertEqual(len(self.reporter.unexpectedSuccesses), 0)
    self.assertEqual(self.reporter.successes, 0)