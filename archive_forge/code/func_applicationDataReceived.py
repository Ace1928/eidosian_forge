from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def applicationDataReceived(self, data):
    """
        Record the given data in C{self.events}.
        """
    self.events.append(('bytes', data))