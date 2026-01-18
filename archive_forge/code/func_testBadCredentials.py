from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
def testBadCredentials(self):
    badCreds = [credentials.UsernamePassword(u, b'wrong password') for u, p in self.users]
    d = defer.DeferredList([self.port.login(c, None, ITestable) for c in badCreds], consumeErrors=True)
    d.addCallback(self._assertFailures, error.UnauthorizedLogin)
    return d