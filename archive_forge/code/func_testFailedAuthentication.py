from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testFailedAuthentication(self):
    self.addPortal()
    self.register()
    self.assertEqual(len(self.registry.users), 0)
    self.assertEqual(len(self.sent), 1)
    dest, m = self.sent[0]
    self.assertEqual(m.code, 401)