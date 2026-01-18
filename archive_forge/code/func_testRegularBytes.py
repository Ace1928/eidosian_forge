from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testRegularBytes(self):
    h = self.p.protocol
    L = [b'here are some bytes la la la', b'some more arrive here', b'lots of bytes to play with', b'la la la', b'ta de da', b'dum']
    for b in L:
        self.p.dataReceived(b)
    self.assertEqual(h.data, b''.join(L))