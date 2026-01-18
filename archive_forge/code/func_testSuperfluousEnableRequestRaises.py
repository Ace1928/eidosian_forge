from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testSuperfluousEnableRequestRaises(self):
    s = self.p.getOptionState(b'\xab')
    s.him.state = 'yes'
    d = self.p.do(b'\xab')
    return self.assertFailure(d, telnet.AlreadyEnabled)