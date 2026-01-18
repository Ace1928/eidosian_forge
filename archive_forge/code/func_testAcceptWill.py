from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testAcceptWill(self):
    cmd = telnet.IAC + telnet.WILL + b'\x91'
    data = b'header' + cmd + b'padding'
    h = self.p.protocol
    h.remoteEnableable = (b'\x91',)
    self.p.dataReceived(data)
    self.assertEqual(self.t.value(), telnet.IAC + telnet.DO + b'\x91')
    self._enabledHelper(h, eR=[b'\x91'])