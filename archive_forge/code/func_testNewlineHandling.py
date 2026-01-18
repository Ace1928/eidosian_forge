from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testNewlineHandling(self):
    h = self.p.protocol
    L = [b'here is the first line\r\n', b'here is the second line\r\x00', b'here is the third line\r\n', b'here is the last line\r\x00']
    for b in L:
        self.p.dataReceived(b)
    self.assertEqual(h.data, L[0][:-2] + b'\n' + L[1][:-2] + b'\r' + L[2][:-2] + b'\n' + L[3][:-2] + b'\r')