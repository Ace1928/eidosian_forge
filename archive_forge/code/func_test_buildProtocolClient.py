import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
def test_buildProtocolClient(self):
    """
        L{ClientFactory.buildProtocol} should be invoked with the address of
        the server to which a connection has been established, which should
        be the same as the address reported by the C{getHost} method of the
        transport of the server protocol and as the C{getPeer} method of the
        transport of the client protocol.
        """
    serverHost = self.server.protocol.transport.getHost()
    clientPeer = self.client.protocol.transport.getPeer()
    self.assertEqual(self.clientWrapper.addresses, [IPv4Address('TCP', serverHost.host, serverHost.port)])
    self.assertEqual(self.clientWrapper.addresses, [IPv4Address('TCP', clientPeer.host, clientPeer.port)])