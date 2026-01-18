from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def test_amoralRPort(self):
    """
        rport is allowed without a value, apparently because server
        implementors might be too stupid to check the received port
        against 5060 and see if they're equal, and because client
        implementors might be too stupid to bind to port 5060, or set a
        value on the rport parameter they send if they bind to another
        port.
        """
    p = self.clientPort.getHost().port
    r = sip.Request('REGISTER', 'sip:bell.example.com')
    r.addHeader('to', 'sip:joe@bell.example.com')
    r.addHeader('contact', 'sip:joe@127.0.0.1:%d' % p)
    r.addHeader('via', sip.Via('127.0.0.1', port=p, rport=True).toString())
    warnings = self.flushWarnings(offendingFunctions=[self.test_amoralRPort])
    self.assertEqual(len(warnings), 1)
    self.assertEqual(warnings[0]['message'], 'rport=True is deprecated since Twisted 9.0.')
    self.assertEqual(warnings[0]['category'], DeprecationWarning)
    self.client.sendMessage(sip.URL(host='127.0.0.1', port=self.serverAddress[1]), r)
    d = self.client.deferred

    def check(received):
        self.assertEqual(len(received), 1)
        r = received[0]
        self.assertEqual(r.code, 200)
    d.addCallback(check)
    return d