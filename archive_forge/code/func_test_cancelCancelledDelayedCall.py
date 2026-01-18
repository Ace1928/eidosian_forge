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
def test_cancelCancelledDelayedCall(self):
    """
        Test that cancelling a DelayedCall which has already been cancelled
        raises the appropriate exception.
        """
    call = reactor.callLater(0, lambda: None)
    call.cancel()
    self.assertRaises(error.AlreadyCancelled, call.cancel)