from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
def testUserLookup(self):
    self.db = checkers.FilePasswordDB(self.dbfile)
    for u, p in self.users:
        self.assertRaises(KeyError, self.db.getUser, u.upper())
        self.assertEqual(self.db.getUser(u), (u, p))