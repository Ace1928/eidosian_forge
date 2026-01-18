from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testWrongToDomainRegister(self):
    r = sip.Request('REGISTER', 'sip:bell.example.com')
    r.addHeader('to', 'sip:joe@foo.com')
    r.addHeader('contact', 'sip:joe@client.com:1234')
    r.addHeader('via', sip.Via('client.com').toString())
    self.proxy.datagramReceived(r.toString(), ('client.com', 5060))
    self.assertEqual(len(self.sent), 0)