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
def test_deferredFailureAfterSuccess(self):
    """
        Check that a successful L{threads.deferToThread} followed by a one
        that raises an exception correctly result as a failure.
        """
    d = threads.deferToThread(lambda: None)
    d.addCallback(lambda ign: threads.deferToThread(lambda: 1 // 0))
    return self.assertFailure(d, ZeroDivisionError)