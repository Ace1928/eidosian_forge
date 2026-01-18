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
def test_cancelCalledDelayedCallAsynchronous(self):
    """
        Test that cancelling a DelayedCall after it has run its function
        raises the appropriate exception.
        """
    d = Deferred()

    def check():
        try:
            self.assertRaises(error.AlreadyCalled, call.cancel)
        except BaseException:
            d.errback()
        else:
            d.callback(None)

    def later():
        reactor.callLater(0, check)
    call = reactor.callLater(0, later)
    return d