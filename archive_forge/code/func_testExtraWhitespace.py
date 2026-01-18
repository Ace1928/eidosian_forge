from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testExtraWhitespace(self):
    v1 = sip.parseViaHeader('SIP/2.0/UDP 192.168.1.1:5060')
    v2 = sip.parseViaHeader('SIP/2.0/UDP     192.168.1.1:5060')
    self.assertEqual(v1.transport, v2.transport)
    self.assertEqual(v1.host, v2.host)
    self.assertEqual(v1.port, v2.port)