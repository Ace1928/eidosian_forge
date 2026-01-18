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
def test_connectDestination(self) -> None:
    """
        L{MemoryReactor.connectTCP}, L{MemoryReactor.connectSSL}, and
        L{MemoryReactor.connectUNIX} will return an L{IConnector} whose
        C{getDestination} method returns an L{IAddress} with attributes which
        reflect the values passed.
        """
    memoryReactor = MemoryReactor()
    for connector in [memoryReactor.connectTCP('test.example.com', 8321, ClientFactory()), memoryReactor.connectSSL('test.example.com', 8321, ClientFactory(), None)]:
        verifyObject(IConnector, connector)
        address = connector.getDestination()
        verifyObject(IAddress, address)
        self.assertEqual(address.host, 'test.example.com')
        self.assertEqual(address.port, 8321)
    connector = memoryReactor.connectUNIX(b'/fake/path', ClientFactory())
    verifyObject(IConnector, connector)
    address = connector.getDestination()
    verifyObject(IAddress, address)
    self.assertEqual(address.name, b'/fake/path')