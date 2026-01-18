from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testIgnoreDo(self):
    cmd = telnet.IAC + telnet.DO + b'V'
    s = self.p.getOptionState(b'V')
    s.us.state = 'yes'
    data = b'tra la la' + cmd + b'dum de dum'
    self.p.dataReceived(data)
    self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
    self.assertEqual(self.t.value(), b'')
    self._enabledHelper(self.p.protocol)