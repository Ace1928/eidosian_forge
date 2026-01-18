from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_canRecurseOverride(self):
    """
        L{server.DNSServerFactory.__init__} sets C{canRecurse} to L{True} if it
        is supplied with C{clients}.
        """
    self.assertEqual(server.DNSServerFactory(clients=[None]).canRecurse, True)