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
def test_cleanupsContinueRunningAfterError(self):
    """
        If a cleanup raises an error then that does not stop the other
        cleanups from being run.
        """
    self.test.addCleanup(self.test.append, 'foo')
    self.test.addCleanup(self.test.fail, 'bar')
    self.test.run(self.result)
    self.assertEqual(['setUp', 'runTest', 'foo', 'tearDown'], self.test.log)
    self.assertEqual(1, len(self.result.errors))
    [(test, error)] = self.result.errors
    self.assertEqual(test, self.test)
    self.assertEqual(error.getErrorMessage(), 'bar')