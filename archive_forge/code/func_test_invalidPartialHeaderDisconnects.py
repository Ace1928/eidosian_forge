from typing import Optional
from unittest import mock
from twisted.internet import address
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.trial import unittest
from .._wrapper import HAProxyWrappingFactory
def test_invalidPartialHeaderDisconnects(self) -> None:
    """
        Test if invalid headers result in connectionLost events.
        """
    factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
    proto = factory.buildProtocol(address.IPv4Address('TCP', '127.1.1.1', 8080))
    transport = StringTransportWithDisconnection()
    transport.protocol = proto
    proto.makeConnection(transport)
    proto.dataReceived(b'PROXY TCP4 1.1.1.1\r\n')
    proto.dataReceived(b'2.2.2.2 8080\r\n')
    self.assertFalse(transport.connected)