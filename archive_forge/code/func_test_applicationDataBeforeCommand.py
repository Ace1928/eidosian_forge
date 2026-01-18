from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_applicationDataBeforeCommand(self):
    """
        Application bytes received before a WILL/WONT/DO/DONT are delivered
        before the command is processed.
        """
    self.protocol.commandMap = {}
    self._deliver(b'y' + telnet.IAC + telnet.WILL + b'\x00', ('bytes', b'y'), ('command', telnet.WILL, b'\x00'))