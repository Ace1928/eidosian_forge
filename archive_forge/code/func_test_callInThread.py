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
def test_callInThread(self):
    """
        Test callInThread functionality: set a C{threading.Event}, and check
        that it's not in the main thread.
        """

    def cb(ign):
        waiter = threading.Event()
        result = []

        def threadedFunc():
            result.append(threadable.isInIOThread())
            waiter.set()
        reactor.callInThread(threadedFunc)
        waiter.wait(120)
        if not waiter.isSet():
            self.fail('Timed out waiting for event.')
        else:
            self.assertEqual(result, [False])
    return self._waitForThread().addCallback(cb)