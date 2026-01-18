import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def cbJoined(ignored):
    d = self.server.packetReceived = Deferred()
    c.transport.write(b'hello world', ('225.0.0.250', addr.port))
    return d