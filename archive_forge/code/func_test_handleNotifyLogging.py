from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_handleNotifyLogging(self):
    """
        L{server.DNSServerFactory.handleNotify} logs the message origin address
        if C{verbose > 0}.
        """
    f = NoResponseDNSServerFactory(verbose=1)
    assertLogMessage(self, ["Notify message from ('::1', 53)"], f.handleNotify, message=dns.Message(), protocol=NoopProtocol(), address=('::1', 53))