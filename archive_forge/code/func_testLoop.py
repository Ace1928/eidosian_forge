from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testLoop(self):
    r = sip.Request('INVITE', 'sip:foo')
    r.addHeader('via', sip.Via('1.2.3.4').toString())
    r.addHeader('via', sip.Via('127.0.0.1').toString())
    self.proxy.datagramReceived(r.toString(), ('client.com', 5060))
    self.assertEqual(self.sent, [])