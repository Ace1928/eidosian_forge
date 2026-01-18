from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_gotResolverResponse(self):
    """
        L{server.DNSServerFactory.gotResolverResponse} accepts a tuple of
        resource record lists and triggers a response message containing those
        resource record lists.
        """
    f = server.DNSServerFactory()
    answers = []
    authority = []
    additional = []
    e = self.assertRaises(RaisingProtocol.WriteMessageArguments, f.gotResolverResponse, (answers, authority, additional), protocol=RaisingProtocol(), message=dns.Message(), address=None)
    (message,), kwargs = e.args
    self.assertIs(message.answers, answers)
    self.assertIs(message.authority, authority)
    self.assertIs(message.additional, additional)