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
def testUNIX(self):
    s = service.MultiService()
    s.startService()
    factory = protocol.ServerFactory()
    factory.protocol = TestEcho
    TestEcho.d = defer.Deferred()
    t = internet.UNIXServer('echo.skt', factory)
    t.setServiceParent(s)
    factory = protocol.ClientFactory()
    factory.protocol = Foo
    factory.d = defer.Deferred()
    factory.line = None
    internet.UNIXClient('echo.skt', factory).setServiceParent(s)
    factory.d.addCallback(self.assertEqual, b'lalala')
    factory.d.addCallback(lambda x: s.stopService())
    factory.d.addCallback(lambda x: TestEcho.d)
    factory.d.addCallback(self._cbTestUnix, factory, s)
    return factory.d