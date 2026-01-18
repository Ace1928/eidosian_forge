import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def setDeferred(self, d):
    """
        Set the Deferred which will be called back when datagramReceived is
        called.
        """
    self.d = d