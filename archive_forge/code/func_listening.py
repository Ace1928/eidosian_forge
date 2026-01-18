import socket
from gc import collect
from typing import Optional
from weakref import ref
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.interfaces import IConnector, IReactorFDSet
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.reactormixins import needsRunningReactor
from twisted.python import context, log
from twisted.python.failure import Failure
from twisted.python.log import ILogContext, err, msg
from twisted.python.runtime import platform
from twisted.test.test_tcp import ClosingProtocol
from twisted.trial.unittest import SkipTest
def listening(port):
    msg(f'Listening on {port.getHost()!r}')
    endpoint = self.endpoints.client(reactor, port.getHost())
    client = endpoint.connect(ClientFactory.forProtocol(lambda: clientProtocol))

    def disconnect(proto):
        msg(f'About to disconnect {proto!r}')
        proto.transport.loseConnection()
    client.addCallback(disconnect)
    client.addErrback(lostConnectionDeferred.errback)
    return lostConnectionDeferred