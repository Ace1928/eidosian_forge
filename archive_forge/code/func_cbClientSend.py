import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def cbClientSend(ignored):
    if clientWrites:
        nextClientWrite = server.packetReceived = defer.Deferred()
        nextClientWrite.addCallback(cbClientSend)
        client.transport.write(*clientWrites.pop(0))
        return nextClientWrite