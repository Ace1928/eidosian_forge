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
def test_deferredFailure(self):
    """
        Check that L{threads.deferToThreadPool} return a failure object with an
        appropriate exception instance when the called function raises an
        exception.
        """

    class NewError(Exception):
        pass

    def raiseError():
        raise NewError()
    d = threads.deferToThreadPool(reactor, self.tp, raiseError)
    return self.assertFailure(d, NewError)