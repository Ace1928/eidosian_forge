from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def connectProxy(self, proxyClient):
    """
        Connect a proxy client to a L{StringTransportWithDisconnection}.

        @param proxyClient: A L{ProxyClient}.
        @return: The L{StringTransportWithDisconnection}.
        """
    clientTransport = StringTransportWithDisconnection()
    clientTransport.protocol = proxyClient
    proxyClient.makeConnection(clientTransport)
    return clientTransport