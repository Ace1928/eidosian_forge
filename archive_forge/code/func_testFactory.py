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
def testFactory(self):
    factory = MyFactory()
    protocol = factory.buildProtocol(None)
    self.assertEqual(protocol.factory, factory)
    self.assertIsInstance(protocol, factory.protocol)