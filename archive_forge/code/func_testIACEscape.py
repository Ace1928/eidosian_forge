from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testIACEscape(self):
    h = self.p.protocol
    L = [b'here are some bytes\xff\xff with an embedded IAC', b'and here is a test of a border escape\xff', b'\xff did you get that IAC?']
    for b in L:
        self.p.dataReceived(b)
    self.assertEqual(h.data, b''.join(L).replace(b'\xff\xff', b'\xff'))