from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testRequestForward(self):
    r = sip.Request('INVITE', 'sip:foo')
    r.addHeader('via', sip.Via('1.2.3.4').toString())
    r.addHeader('via', sip.Via('1.2.3.5').toString())
    r.addHeader('foo', 'bar')
    r.addHeader('to', '<sip:joe@server.com>')
    r.addHeader('contact', '<sip:joe@1.2.3.5>')
    self.proxy.datagramReceived(r.toString(), ('1.2.3.4', 5060))
    self.assertEqual(len(self.sent), 1)
    dest, m = self.sent[0]
    self.assertEqual(dest.port, 5060)
    self.assertEqual(dest.host, 'server.com')
    self.assertEqual(m.uri.toString(), 'sip:foo')
    self.assertEqual(m.method, 'INVITE')
    self.assertEqual(m.headers['via'], ['SIP/2.0/UDP 127.0.0.1:5060', 'SIP/2.0/UDP 1.2.3.4:5060', 'SIP/2.0/UDP 1.2.3.5:5060'])