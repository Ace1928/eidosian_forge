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
def test_callLaterReset(self):
    """
        A L{DelayedCall} that is reset will be scheduled at the new time.
        """
    call = reactor.callLater(2, passthru, passthru)
    self.addCleanup(call.cancel)
    origTime = call.time
    call.reset(1)
    self.assertNotEqual(call.time, origTime)