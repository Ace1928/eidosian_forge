import socket
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import defer, error
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred, maybeDeferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import DatagramProtocol
from twisted.internet.test.connectionmixins import LogObserverMixin, findFreePort
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python import context
from twisted.python.log import ILogContext, err
from twisted.test.test_udp import GoodClient, Server
from twisted.trial.unittest import SkipTest
@skipWithoutIPv6
def test_connectedWriteToIPv6Interface(self):
    """
        An IPv6 address can be passed as the C{interface} argument to
        L{listenUDP}. The resulting Port accepts IPv6 datagrams.
        """
    reactor = self.buildReactor()
    server = Server()
    serverStarted = server.startedDeferred = defer.Deferred()
    self.getListeningPort(reactor, server, interface='::1')
    client = GoodClient()
    clientStarted = client.startedDeferred = defer.Deferred()
    self.getListeningPort(reactor, client, interface='::1')
    cAddr = client.transport.getHost()

    def cbClientStarted(ignored):
        """
            Send a datagram from the client once it's started.

            @param ignored: a list of C{[None, None]}, which is ignored
            @returns: a deferred which fires when the server has received a
                datagram.
            """
        client.transport.connect('::1', server.transport.getHost().port)
        client.transport.write(b'spam')
        serverReceived = server.packetReceived = defer.Deferred()
        return serverReceived

    def cbServerReceived(ignored):
        """
            Stop the reactor after a datagram is received.

            @param ignored: L{None}, which is ignored
            @returns: L{None}
            """
        reactor.stop()
    d = defer.gatherResults([serverStarted, clientStarted])
    d.addCallback(cbClientStarted)
    d.addCallback(cbServerReceived)
    d.addErrback(err)
    self.runReactor(reactor)
    packet = server.packets[0]
    self.assertEqual(packet, (b'spam', (cAddr.host, cAddr.port)))