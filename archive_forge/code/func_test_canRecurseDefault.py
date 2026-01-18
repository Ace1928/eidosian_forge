from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_canRecurseDefault(self):
    """
        L{server.DNSServerFactory.canRecurse} is a flag indicating that this
        server is capable of performing recursive DNS lookups. It defaults to
        L{False}.
        """
    self.assertFalse(server.DNSServerFactory().canRecurse)