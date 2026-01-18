from __future__ import annotations
from typing import Callable
from zope.interface.verify import verifyObject
from typing_extensions import Protocol
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory
from twisted.internet.testing import (
from twisted.python.reflect import namedAny
from twisted.trial.unittest import TestCase
def test_listenDefaultHost(self) -> None:
    """
        L{MemoryReactor.listenTCP}, L{MemoryReactor.listenSSL} and
        L{MemoryReactor.listenUNIX} will return an L{IListeningPort} whose
        C{getHost} method returns an L{IAddress}; C{listenTCP} and C{listenSSL}
        will have a default host of C{'0.0.0.0'}, and a port that reflects the
        value passed, and C{listenUNIX} will have a name that reflects the path
        passed.
        """
    memoryReactor = MemoryReactor()
    for port in [memoryReactor.listenTCP(8242, Factory()), memoryReactor.listenSSL(8242, Factory(), None)]:
        verifyObject(IListeningPort, port)
        address = port.getHost()
        verifyObject(IAddress, address)
        self.assertEqual(address.host, '0.0.0.0')
        self.assertEqual(address.port, 8242)
    port = memoryReactor.listenUNIX(b'/path/to/socket', Factory())
    verifyObject(IListeningPort, port)
    address = port.getHost()
    verifyObject(IAddress, address)
    self.assertEqual(address.name, b'/path/to/socket')