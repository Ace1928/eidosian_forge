import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
def test_lotsOfThreadsAreScheduledCorrectly(self):
    """
        L{IReactorThreads.callFromThread} can be used to schedule a large
        number of calls in the reactor thread.
        """

    def addAndMaybeFinish():
        self.counter += 1
        if self.counter == 100:
            self.deferred.callback(True)
    for i in range(100):
        self.schedule(addAndMaybeFinish)
    return self.deferred