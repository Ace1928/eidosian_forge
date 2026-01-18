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
def test_leftoverPendingCalls(self):
    """
        Trial reports a L{util.DirtyReactorAggregateError} and fails the test
        if a test leaves a L{DelayedCall} hanging.
        """
    suite = erroneous.ReactorCleanupTests('test_leftoverPendingCalls')
    suite.run(self.result)
    self.assertFalse(self.result.wasSuccessful())
    failure = self.result.errors[0][1]
    self.assertEqual(self.result.successes, 0)
    self.assertTrue(failure.check(util.DirtyReactorAggregateError))