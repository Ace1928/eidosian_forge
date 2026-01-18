from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_gotResolverErrorResetsResponseAttributes(self):
    """
        L{server.DNSServerFactory.gotResolverError} does not allow request
        attributes to leak into the response ie it sends a response with AD, CD
        set to 0 and empty response record sections.
        """
    factory = server.DNSServerFactory()
    responses = []
    factory.sendReply = lambda protocol, response, address: responses.append(response)
    request = dns.Message(authenticData=True, checkingDisabled=True)
    request.answers = [object(), object()]
    request.authority = [object(), object()]
    request.additional = [object(), object()]
    factory.gotResolverError(failure.Failure(error.DomainError()), protocol=None, message=request, address=None)
    self.assertEqual([dns.Message(rCode=3, answer=True)], responses)