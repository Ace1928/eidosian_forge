import os
import socket
import sys
from unittest import skipIf
from twisted.internet import address, defer, error, interfaces, protocol, reactor, utils
from twisted.python import lockfile
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.test.test_tcp import MyClientFactory, MyServerFactory
from twisted.trial import unittest
def test_dumber(self):
    """
        L{IReactorUNIX.connectUNIX} can be used to connect a client to a server
        started with L{IReactorUNIX.listenUNIX}.
        """
    filename = self.mktemp()
    serverFactory = MyServerFactory()
    serverConnMade = defer.Deferred()
    serverFactory.protocolConnectionMade = serverConnMade
    unixPort = reactor.listenUNIX(filename, serverFactory)
    self.addCleanup(unixPort.stopListening)
    clientFactory = MyClientFactory()
    clientConnMade = defer.Deferred()
    clientFactory.protocolConnectionMade = clientConnMade
    reactor.connectUNIX(filename, clientFactory)
    d = defer.gatherResults([serverConnMade, clientConnMade])

    def allConnected(args):
        serverProtocol, clientProtocol = args
        self.assertEqual(clientFactory.peerAddresses, [address.UNIXAddress(filename)])
        clientProtocol.transport.loseConnection()
        serverProtocol.transport.loseConnection()
    d.addCallback(allConnected)
    return d