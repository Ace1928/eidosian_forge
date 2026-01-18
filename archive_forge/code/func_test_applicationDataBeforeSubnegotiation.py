from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_applicationDataBeforeSubnegotiation(self):
    """
        Application bytes received before a subnegotiation command are
        delivered before the negotiation is processed.
        """
    self._deliver(b'z' + telnet.IAC + telnet.SB + b'Qx' + telnet.IAC + telnet.SE, ('bytes', b'z'), ('negotiate', b'Q', [b'x']))