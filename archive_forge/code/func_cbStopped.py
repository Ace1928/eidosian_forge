import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def cbStopped(ignored):
    d = server.startedDeferred = defer.Deferred()
    p = reactor.listenUDP(0, server, interface='127.0.0.1')
    return d.addCallback(cbStarted, p)