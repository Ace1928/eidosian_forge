from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_disableRemote(self):
    """
        It is an error for L{telnet.Telnet.disableRemote} to be called, since
        L{telnet.Telnet.enableRemote} will never allow any options to be
        enabled remotely.  If a subclass overrides enableRemote, it must also
        override disableRemote.
        """
    self.assertRaises(NotImplementedError, self.protocol.disableRemote, b'\x00')