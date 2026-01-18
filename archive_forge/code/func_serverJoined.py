import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def serverJoined(ignored):
    d1 = firstClient.packetReceived = Deferred()
    d2 = secondClient.packetReceived = Deferred()
    firstClient.transport.write(b'hello world', (theGroup, portno))
    return gatherResults([d1, d2])