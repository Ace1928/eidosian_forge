from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_escapedControl(self):
    """
        IAC in the escaped state gets delivered and so does another
        application-data byte following it.
        """
    self._deliver(telnet.IAC)
    self._deliver(telnet.IAC + b'g', ('bytes', telnet.IAC + b'g'))