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
@skipIf(not interfaces.IReactorUNIX(reactor, None), 'This reactor does not support UNIX domain sockets')
def testStoppingServer(self):
    factory = protocol.ServerFactory()
    factory.protocol = wire.Echo
    t = internet.UNIXServer('echo.skt', factory)
    t.startService()
    t.stopService()
    self.assertFalse(t.running)
    factory = protocol.ClientFactory()
    d = defer.Deferred()
    factory.clientConnectionFailed = lambda *args: d.callback(None)
    reactor.connectUNIX('echo.skt', factory)
    return d