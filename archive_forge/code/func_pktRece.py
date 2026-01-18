import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def pktRece(ignored):
    self.server.transport.connectionLost()
    reactor.callLater(0, finished.callback, None)