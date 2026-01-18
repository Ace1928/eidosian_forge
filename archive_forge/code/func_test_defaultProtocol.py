from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_defaultProtocol(self):
    """
        L{server.DNSServerFactory.protocol} defaults to L{dns.DNSProtocol}.
        """
    self.assertIs(server.DNSServerFactory.protocol, dns.DNSProtocol)