from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
def test_getUserNonexistentDatabase(self):
    """
        A missing db file will cause a permanent rejection of authorization
        attempts.
        """
    self.db = checkers.FilePasswordDB('test_thisbetternoteverexist.db')
    self.assertRaises(error.UnauthorizedLogin, self.db.getUser, 'user')