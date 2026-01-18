from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_applicationDataBeforeSimpleCommand(self):
    """
        Application bytes received before a command are delivered before the
        command is processed.
        """
    self._deliver(b'x' + telnet.IAC + telnet.NOP, ('bytes', b'x'), ('command', telnet.NOP, None))