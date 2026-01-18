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
def test_callLaterUsesReactorSecondsInDelayedCall(self):
    """
        L{reactor.callLater<twisted.internet.interfaces.IReactorTime.callLater>}
        should use the reactor's seconds factory
        to produce the time at which the DelayedCall will be called.
        """
    oseconds = reactor.seconds
    reactor.seconds = lambda: 100
    try:
        call = reactor.callLater(5, lambda: None)
        self.assertEqual(call.getTime(), 105)
    finally:
        reactor.seconds = oseconds
    call.cancel()