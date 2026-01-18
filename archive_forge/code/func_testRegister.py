from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testRegister(self):
    p = self.clientPort.getHost().port
    r = sip.Request('REGISTER', 'sip:bell.example.com')
    r.addHeader('to', 'sip:joe@bell.example.com')
    r.addHeader('contact', 'sip:joe@127.0.0.1:%d' % p)
    r.addHeader('via', sip.Via('127.0.0.1', port=p).toString())
    self.client.sendMessage(sip.URL(host='127.0.0.1', port=self.serverAddress[1]), r)
    d = self.client.deferred

    def check(received):
        self.assertEqual(len(received), 1)
        r = received[0]
        self.assertEqual(r.code, 200)
    d.addCallback(check)
    return d