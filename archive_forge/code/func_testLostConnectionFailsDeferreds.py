from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testLostConnectionFailsDeferreds(self):
    d1 = self.p.do(b'\x12')
    d2 = self.p.do(b'#')
    d3 = self.p.do(b'4')

    class TestException(Exception):
        pass
    self.p.connectionLost(TestException('Total failure!'))
    d1 = self.assertFailure(d1, TestException)
    d2 = self.assertFailure(d2, TestException)
    d3 = self.assertFailure(d3, TestException)
    return defer.gatherResults([d1, d2, d3])