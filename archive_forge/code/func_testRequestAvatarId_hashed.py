from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
def testRequestAvatarId_hashed(self):
    self.db = checkers.FilePasswordDB(self.dbfile)
    UsernameHashedPassword = self.getDeprecatedModuleAttribute('twisted.cred.credentials', 'UsernameHashedPassword', _uhpVersion)
    creds = [UsernameHashedPassword(u, p) for u, p in self.users]
    d = defer.gatherResults([defer.maybeDeferred(self.db.requestAvatarId, c) for c in creds])
    d.addCallback(self.assertEqual, [u for u, p in self.users])
    return d