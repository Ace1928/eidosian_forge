from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_gotResolverErrorLogging(self):
    """
        L{server.DNSServerFactory.gotResolver} logs a message if C{verbose > 0}.
        """
    f = NoResponseDNSServerFactory(verbose=1)
    assertLogMessage(self, ['Lookup failed'], f.gotResolverError, failure.Failure(error.DomainError()), protocol=NoopProtocol(), message=dns.Message(), address=None)