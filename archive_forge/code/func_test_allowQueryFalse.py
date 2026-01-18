from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_allowQueryFalse(self):
    """
        If C{allowQuery} returns C{False},
        L{server.DNSServerFactory.messageReceived} calls L{server.sendReply}
        with a message whose C{rCode} is L{dns.EREFUSED}.
        """

    class SendReplyException(Exception):
        pass

    class RaisingDNSServerFactory(server.DNSServerFactory):

        def allowQuery(self, *args, **kwargs):
            return False

        def sendReply(self, *args, **kwargs):
            raise SendReplyException(args, kwargs)
    f = RaisingDNSServerFactory()
    e = self.assertRaises(SendReplyException, f.messageReceived, message=dns.Message(), proto=None, address=None)
    (proto, message, address), kwargs = e.args
    self.assertEqual(message.rCode, dns.EREFUSED)