from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testResponseToUs(self):
    r = sip.Response(200)
    r.addHeader('via', sip.Via('127.0.0.1').toString())
    l = []
    self.proxy.gotResponse = lambda *a: l.append(a)
    self.proxy.datagramReceived(r.toString(), ('1.1.1.1', 5060))
    self.assertEqual(len(l), 1)
    m, addr = l[0]
    self.assertEqual(len(m.headers.get('via', [])), 0)
    self.assertEqual(m.code, 200)