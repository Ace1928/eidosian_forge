from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
def testHashedCredentials(self):
    UsernameHashedPassword = self.getDeprecatedModuleAttribute('twisted.cred.credentials', 'UsernameHashedPassword', _uhpVersion)
    hashedCreds = [UsernameHashedPassword(u, self.hash(None, p, u[:2])) for u, p in self.users]
    d = defer.DeferredList([self.port.login(c, None, ITestable) for c in hashedCreds], consumeErrors=True)
    d.addCallback(self._assertFailures, error.UnhandledCredentials)
    return d