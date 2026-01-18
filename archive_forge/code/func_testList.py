import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def testList(self):
    user = self._loggedInUser('someuser')
    user.transport.clear()
    somegroup = self.successResultOf(self.realm.createGroup('somegroup'))
    somegroup.size = lambda: succeed(17)
    somegroup.meta['topic'] = 'this is the topic woo'
    user.write('LIST #somegroup\r\n')
    r = self._response(user)
    self.assertEqual(len(r), 2)
    resp, end = r
    self.assertEqual(resp[0], 'realmname')
    self.assertEqual(resp[1], '322')
    self.assertEqual(resp[2][0], 'someuser')
    self.assertEqual(resp[2][1], 'somegroup')
    self.assertEqual(resp[2][2], '17')
    self.assertEqual(resp[2][3], 'this is the topic woo')
    self.assertEqual(end[0], 'realmname')
    self.assertEqual(end[1], '323')
    self.assertEqual(end[2][0], 'someuser')
    self.assertEqual(end[2][1], 'End of /LIST')
    user.transport.clear()
    user.write('LIST\r\n')
    r = self._response(user)
    self.assertEqual(len(r), 2)
    fg1, end = r
    self.assertEqual(fg1[1], '322')
    self.assertEqual(fg1[2][1], 'somegroup')
    self.assertEqual(fg1[2][2], '17')
    self.assertEqual(fg1[2][3], 'this is the topic woo')
    self.assertEqual(end[1], '323')