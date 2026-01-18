from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_gotResolverErrorAuthoritativeDomainError(self):
    """
        L{server.DNSServerFactory.gotResolver} triggers a response message with
        an C{rCode} of L{dns.ENAME} if supplied with a
        L{error.AuthoritativeDomainError}.
        """
    self._assertMessageRcodeForError(error.AuthoritativeDomainError(), dns.ENAME)