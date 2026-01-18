from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_messageReceivedTimestamp(self):
    """
        L{server.DNSServerFactory.messageReceived} assigns a unix timestamp to
        the received message.
        """
    m = dns.Message()
    f = NoResponseDNSServerFactory()
    t = object()
    self.patch(server.time, 'time', lambda: t)
    f.messageReceived(message=m, proto=None, address=None)
    self.assertEqual(m.timeReceived, t)