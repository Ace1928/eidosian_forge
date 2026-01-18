from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_gotResolverResponseLogging(self):
    """
        L{server.DNSServerFactory.gotResolverResponse} logs the total number of
        records in the response if C{verbose > 0}.
        """
    f = NoResponseDNSServerFactory(verbose=1)
    answers = [dns.RRHeader()]
    authority = [dns.RRHeader()]
    additional = [dns.RRHeader()]
    assertLogMessage(self, ['Lookup found 3 records'], f.gotResolverResponse, (answers, authority, additional), protocol=NoopProtocol(), message=dns.Message(), address=None)