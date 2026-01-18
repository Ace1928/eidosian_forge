from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_messageReceivedAllowQuery(self):
    """
        L{server.DNSServerFactory.messageReceived} passes all messages to
        L{server.DNSServerFactory.allowQuery} along with the receiving protocol
        and origin address.
        """
    message = dns.Message()
    dummyProtocol = object()
    dummyAddress = object()
    f = RaisingDNSServerFactory()
    e = self.assertRaises(RaisingDNSServerFactory.AllowQueryArguments, f.messageReceived, message=message, proto=dummyProtocol, address=dummyAddress)
    args, kwargs = e.args
    self.assertEqual(args, (message, dummyProtocol, dummyAddress))
    self.assertEqual(kwargs, {})