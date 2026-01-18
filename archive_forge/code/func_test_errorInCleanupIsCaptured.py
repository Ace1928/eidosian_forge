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
def test_errorInCleanupIsCaptured(self):
    """
        Errors raised in cleanup functions should be treated like errors in
        C{tearDown}. They should be added as errors and fail the test. Skips,
        todos and failures are all treated as errors.
        """
    self.test.addCleanup(self.test.fail, 'foo')
    self.test.run(self.result)
    self.assertFalse(self.result.wasSuccessful())
    self.assertEqual(1, len(self.result.errors))
    [(test, error)] = self.result.errors
    self.assertEqual(test, self.test)
    self.assertEqual(error.getErrorMessage(), 'foo')