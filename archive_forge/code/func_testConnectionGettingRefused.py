import copy
import os
import pickle
from io import StringIO
from unittest import skipIf
from twisted.application import app, internet, reactors, service
from twisted.application.internet import backoffPolicy
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.testing import MemoryReactor
from twisted.persisted import sob
from twisted.plugins import twisted_reactors
from twisted.protocols import basic, wire
from twisted.python import usage
from twisted.python.runtime import platformType
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SkipTest, TestCase
def testConnectionGettingRefused(self):
    factory = protocol.ServerFactory()
    factory.protocol = wire.Echo
    t = internet.TCPServer(0, factory)
    t.startService()
    num = t._port.getHost().port
    t.stopService()
    d = defer.Deferred()
    factory = protocol.ClientFactory()
    factory.clientConnectionFailed = lambda *args: d.callback(None)
    c = internet.TCPClient('127.0.0.1', num, factory)
    c.startService()
    return d