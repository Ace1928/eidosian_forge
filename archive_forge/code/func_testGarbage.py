from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testGarbage(self):
    l = self.l
    self.feedMessage(request3)
    self.assertEqual(len(l), 1)
    self.validateMessage(l[0], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe'], 'content-length': ['4']}, '1234')