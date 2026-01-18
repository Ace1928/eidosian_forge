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
def test_SRVNoResult(self):
    """
        Test connectTCP gets called with fallback parameters on empty result.
        """
    client.theResolver.results = []
    self.connector.connect()
    self.assertIsNone(self.factory.reason)
    self.assertEqual(self.reactor.tcpClients.pop()[:2], ('example.org', 'xmpp-server'))