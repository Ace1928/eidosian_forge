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
def testCallFromThreadStops(self):
    """
        Ensure that callFromThread from inside a callFromThread
        callback doesn't sit in an infinite loop and lets other
        things happen too.
        """
    self.stopped = False
    d = defer.Deferred()
    reactor.callFromThread(self._callFromThreadCallback, d)
    return d