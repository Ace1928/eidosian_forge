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
def test_addCleanupWaitsForDeferreds(self):
    """
        If an added callable returns a L{Deferred}, then the test should wait
        until that L{Deferred} has fired before running the next cleanup
        method.
        """

    def cleanup(message):
        d = defer.Deferred()
        reactor.callLater(0, d.callback, message)
        return d.addCallback(self.test.append)
    self.test.addCleanup(self.test.append, 'foo')
    self.test.addCleanup(cleanup, 'bar')
    self.test.run(self.result)
    self.assertEqual(['setUp', 'runTest', 'bar', 'foo', 'tearDown'], self.test.log)