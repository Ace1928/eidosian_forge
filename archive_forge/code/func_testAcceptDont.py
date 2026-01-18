from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testAcceptDont(self):
    cmd = telnet.IAC + telnet.DONT + b')'
    s = self.p.getOptionState(b')')
    s.us.state = 'yes'
    data = b'fiddle dum ' + cmd
    self.p.dataReceived(data)
    self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
    self.assertEqual(self.t.value(), telnet.IAC + telnet.WONT + b')')
    self.assertEqual(s.us.state, 'no')
    self._enabledHelper(self.p.protocol, dL=[b')'])