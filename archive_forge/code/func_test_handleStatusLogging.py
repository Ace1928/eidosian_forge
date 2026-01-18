from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_handleStatusLogging(self):
    """
        L{server.DNSServerFactory.handleStatus} logs the message origin address
        if C{verbose > 0}.
        """
    f = NoResponseDNSServerFactory(verbose=1)
    assertLogMessage(self, ["Status request from ('::1', 53)"], f.handleStatus, message=dns.Message(), protocol=NoopProtocol(), address=('::1', 53))