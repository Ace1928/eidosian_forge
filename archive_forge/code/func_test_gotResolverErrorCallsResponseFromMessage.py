from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_gotResolverErrorCallsResponseFromMessage(self):
    """
        L{server.DNSServerFactory.gotResolverError} calls
        L{server.DNSServerFactory._responseFromMessage} to generate a response.
        """
    factory = NoResponseDNSServerFactory()
    factory._responseFromMessage = raiser
    request = dns.Message()
    request.timeReceived = 1
    e = self.assertRaises(RaisedArguments, factory.gotResolverError, failure.Failure(error.DomainError()), protocol=None, message=request, address=None)
    self.assertEqual(((), dict(message=request, rCode=dns.ENAME)), (e.args, e.kwargs))