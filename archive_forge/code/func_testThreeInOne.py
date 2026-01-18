from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def testThreeInOne(self):
    l = self.l
    self.feedMessage(request4)
    self.assertEqual(len(l), 3)
    self.validateMessage(l[0], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe'], 'content-length': ['0']}, '')
    self.validateMessage(l[1], 'INVITE', 'sip:loop', {'from': ['foo'], 'to': ['bar'], 'content-length': ['4']}, 'abcd')
    self.validateMessage(l[2], 'INVITE', 'sip:loop', {'from': ['foo'], 'to': ['bar'], 'content-length': ['4']}, '1234')