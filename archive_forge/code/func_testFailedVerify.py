import os
import hamcrest
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import waitUntilAllDisconnected
from twisted.protocols import basic
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test.test_tcp import ProperlyCloseFilesMixin
from twisted.trial.unittest import TestCase
from zope.interface import implementer
def testFailedVerify(self):
    org = 'twisted.test.test_ssl'
    self.setupServerAndClient((org, org + ', client'), {}, (org, org + ', server'), {})

    def verify(*a):
        return False
    self.clientCtxFactory.getContext().set_verify(SSL.VERIFY_PEER, verify)
    serverConnLost = defer.Deferred()
    serverProtocol = protocol.Protocol()
    serverProtocol.connectionLost = serverConnLost.callback
    serverProtocolFactory = protocol.ServerFactory()
    serverProtocolFactory.protocol = lambda: serverProtocol
    self.serverPort = serverPort = reactor.listenSSL(0, serverProtocolFactory, self.serverCtxFactory)
    clientConnLost = defer.Deferred()
    clientProtocol = protocol.Protocol()
    clientProtocol.connectionLost = clientConnLost.callback
    clientProtocolFactory = protocol.ClientFactory()
    clientProtocolFactory.protocol = lambda: clientProtocol
    reactor.connectSSL('127.0.0.1', serverPort.getHost().port, clientProtocolFactory, self.clientCtxFactory)
    dl = defer.DeferredList([serverConnLost, clientConnLost], consumeErrors=True)
    return dl.addCallback(self._cbLostConns)