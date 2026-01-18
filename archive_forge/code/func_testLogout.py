import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def testLogout(self):
    logout = []
    firstuser = self.successResultOf(self.realm.lookupUser('firstuser'))
    user = TestCaseUserAgg(firstuser, self.realm, self.factory)
    self._login(user, 'firstuser')
    user.protocol.logout = lambda: logout.append(True)
    user.write('QUIT\r\n')
    self.assertEqual(logout, [True])