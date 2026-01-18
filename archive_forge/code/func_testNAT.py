from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testNAT(self):
    s = 'SIP/2.0/UDP 10.0.0.1:5060;received=22.13.1.5;rport=12345'
    v = sip.parseViaHeader(s)
    self.assertEqual(v.transport, 'UDP')
    self.assertEqual(v.host, '10.0.0.1')
    self.assertEqual(v.port, 5060)
    self.assertEqual(v.received, '22.13.1.5')
    self.assertEqual(v.rport, 12345)
    self.assertNotEqual(v.toString().find('rport=12345'), -1)