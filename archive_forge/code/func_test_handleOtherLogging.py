from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_handleOtherLogging(self):
    """
        L{server.DNSServerFactory.handleOther} logs the message origin address
        if C{verbose > 0}.
        """
    f = NoResponseDNSServerFactory(verbose=1)
    assertLogMessage(self, ["Unknown op code (0) from ('::1', 53)"], f.handleOther, message=dns.Message(), protocol=NoopProtocol(), address=('::1', 53))