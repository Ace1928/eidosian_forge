from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testResponseForward(self):
    r = sip.Response(200)
    r.addHeader('via', sip.Via('127.0.0.1').toString())
    r.addHeader('via', sip.Via('client.com', port=1234).toString())
    self.proxy.datagramReceived(r.toString(), ('1.1.1.1', 5060))
    self.assertEqual(len(self.sent), 1)
    dest, m = self.sent[0]
    self.assertEqual((dest.host, dest.port), ('client.com', 1234))
    self.assertEqual(m.code, 200)
    self.assertEqual(m.headers['via'], ['SIP/2.0/UDP client.com:1234'])