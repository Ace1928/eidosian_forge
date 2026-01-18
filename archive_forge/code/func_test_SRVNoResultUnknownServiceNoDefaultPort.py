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
def test_SRVNoResultUnknownServiceNoDefaultPort(self):
    """
        Connect fails on no result, unknown service and no default port.
        """
    self.connector = srvconnect.SRVConnector(self.reactor, 'thisbetternotexist', 'example.org', self.factory)
    client.theResolver.failure = ServiceNameUnknownError()
    self.connector.connect()
    self.assertTrue(self.factory.reason.check(ServiceNameUnknownError))