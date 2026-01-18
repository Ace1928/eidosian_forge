import errno
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import defer
from twisted.internet.error import CannotListenError, ConnectionRefusedError
from twisted.internet.interfaces import IResolver
from twisted.internet.task import Clock
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.names import cache, client, dns, error, hosts
from twisted.names.common import ResolverBase
from twisted.names.error import DNSQueryTimeoutError
from twisted.names.test import test_util
from twisted.names.test.test_hosts import GoodTempPathMixin
from twisted.python import failure
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_datagramQueryServerOrder(self):
    """
        L{client.Resolver.queryUDP} should issue queries to its
        L{dns.DNSDatagramProtocol} with server addresses taken from its own
        C{servers} and C{dynServers} lists, proceeding through them in order
        as L{DNSQueryTimeoutError}s occur.
        """
    protocol = StubDNSDatagramProtocol()
    servers = [('::1', 53), ('::2', 53)]
    dynServers = [('::3', 53), ('::4', 53)]
    resolver = client.Resolver(servers=servers)
    resolver.dynServers = dynServers
    resolver._connectedProtocol = lambda interface: protocol
    expectedResult = object()
    queryResult = resolver.queryUDP(None)
    queryResult.addCallback(self.assertEqual, expectedResult)
    self.assertEqual(len(protocol.queries), 1)
    self.assertIs(protocol.queries[0][0], servers[0])
    protocol.queries[0][-1].errback(DNSQueryTimeoutError(0))
    self.assertEqual(len(protocol.queries), 2)
    self.assertIs(protocol.queries[1][0], servers[1])
    protocol.queries[1][-1].errback(DNSQueryTimeoutError(1))
    self.assertEqual(len(protocol.queries), 3)
    self.assertIs(protocol.queries[2][0], dynServers[0])
    protocol.queries[2][-1].errback(DNSQueryTimeoutError(2))
    self.assertEqual(len(protocol.queries), 4)
    self.assertIs(protocol.queries[3][0], dynServers[1])
    protocol.queries[3][-1].callback(expectedResult)
    return queryResult