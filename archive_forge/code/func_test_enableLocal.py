from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_enableLocal(self):
    """
        L{telnet.Telnet.enableLocal} should reject all options, since
        L{telnet.Telnet} does not know how to implement any options.
        """
    self.assertFalse(self.protocol.enableLocal(b'\x00'))