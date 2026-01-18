from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testWrongDomainLookup(self):
    self.register()
    url = sip.URL(username='joe', host='foo.com')
    d = self.proxy.locator.getAddress(url)
    self.assertFailure(d, LookupError)
    return d