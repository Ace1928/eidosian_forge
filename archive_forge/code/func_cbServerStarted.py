import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def cbServerStarted(ignored):
    self.port2 = reactor.listenUDP(0, client, interface='127.0.0.1')
    return clientStarted