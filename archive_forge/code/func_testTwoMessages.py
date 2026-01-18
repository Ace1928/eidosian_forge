from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testTwoMessages(self):
    l = self.l
    self.feedMessage(request1)
    self.feedMessage(request2)
    self.assertEqual(len(l), 2)
    self.validateMessage(l[0], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe'], 'content-length': ['4']}, 'abcd')
    self.validateMessage(l[1], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe']}, '1234')