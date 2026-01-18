import random
from zope.interface.verify import verifyObject
from twisted.internet import defer, protocol
from twisted.internet.error import DNSLookupError, ServiceNameUnknownError
from twisted.internet.interfaces import IConnector
from twisted.internet.testing import MemoryReactor
from twisted.names import client, dns, srvconnect
from twisted.names.common import ResolverBase
from twisted.names.error import DNSNameError
from twisted.trial import unittest
def test_SRVPresent(self):
    """
        Test connectTCP gets called with the address from the SRV record.
        """
    payload = dns.Record_SRV(port=6269, target='host.example.org', ttl=60)
    client.theResolver.results = [dns.RRHeader(name='example.org', type=dns.SRV, cls=dns.IN, ttl=60, payload=payload)]
    self.connector.connect()
    self.assertIsNone(self.factory.reason)
    self.assertEqual(self.reactor.tcpClients.pop()[:2], ('host.example.org', 6269))