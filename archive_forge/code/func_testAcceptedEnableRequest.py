from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testAcceptedEnableRequest(self):
    d = self.p.do(b'B')
    h = self.p.protocol
    h.remoteEnableable = (b'B',)
    self.assertEqual(self.t.value(), telnet.IAC + telnet.DO + b'B')
    self.p.dataReceived(telnet.IAC + telnet.WILL + b'B')
    d.addCallback(self.assertEqual, True)
    d.addCallback(lambda _: self._enabledHelper(h, eR=[b'B']))
    return d