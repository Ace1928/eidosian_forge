from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_messageReceivedLogging1(self):
    """
        L{server.DNSServerFactory.messageReceived} logs the query types of all
        queries in the message if C{verbose} is set to C{1}.
        """
    m = dns.Message()
    m.addQuery(name='example.com', type=dns.MX)
    m.addQuery(name='example.com', type=dns.AAAA)
    f = NoResponseDNSServerFactory(verbose=1)
    assertLogMessage(self, ["MX AAAA query from ('192.0.2.100', 53)"], f.messageReceived, message=m, proto=None, address=('192.0.2.100', 53))