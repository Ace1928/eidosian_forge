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
def test_blockingCallFromThread(self):
    """
        Test blockingCallFromThread facility: create a thread, call a function
        in the reactor using L{threads.blockingCallFromThread}, and verify the
        result returned.
        """

    def reactorFunc():
        return defer.succeed('foo')

    def cb(res):
        self.assertEqual(res[0][0], 'foo')
    return self._testBlockingCallFromThread(reactorFunc).addCallback(cb)