import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def gotPackets(ignored):
    self.assertEqual(firstClient.packets[0][0], b'hello world')
    self.assertEqual(secondClient.packets[0][0], b'hello world')