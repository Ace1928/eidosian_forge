import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def testTTL(self):
    for o in (self.client, self.server):
        self.assertEqual(o.transport.getTTL(), 1)
        o.transport.setTTL(2)
        self.assertEqual(o.transport.getTTL(), 2)