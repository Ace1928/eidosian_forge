import os
import sys
import time
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, threads
from twisted.python import failure, log, threadable, threadpool
from twisted.trial.unittest import TestCase
import time
import %(reactor)s
from twisted.internet import reactor
def test_errorBlockingCallFromThread(self):
    """
        Test error report for blockingCallFromThread.
        """

    def reactorFunc():
        return defer.fail(RuntimeError('bar'))

    def cb(res):
        self.assertIsInstance(res[1][0], RuntimeError)
        self.assertEqual(res[1][0].args[0], 'bar')
    return self._testBlockingCallFromThread(reactorFunc).addCallback(cb)