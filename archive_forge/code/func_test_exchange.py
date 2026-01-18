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
def test_exchange(self):
    """
        Test that a datagram can be sent to and received by a server and vice
        versa.
        """
    clientaddr = self.mktemp()
    serveraddr = self.mktemp()
    sp = ServerProto()
    cp = ClientProto()
    s = reactor.listenUNIXDatagram(serveraddr, sp)
    self.addCleanup(s.stopListening)
    c = reactor.connectUNIXDatagram(serveraddr, cp, bindAddress=clientaddr)
    self.addCleanup(c.stopListening)
    d = defer.gatherResults([sp.deferredStarted, cp.deferredStarted])

    def write(ignored):
        cp.transport.write(b'hi')
        return defer.gatherResults([sp.deferredGotWhat, cp.deferredGotBack])

    def _cbTestExchange(ignored):
        self.assertEqual(b'hi', sp.gotwhat)
        self.assertEqual(clientaddr, sp.gotfrom)
        self.assertEqual(b'hi back', cp.gotback)
    d.addCallback(write)
    d.addCallback(_cbTestExchange)
    return d