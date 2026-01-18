import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def testFailedLogin(self):
    firstuser = self.successResultOf(self.realm.lookupUser('firstuser'))
    user = TestCaseUserAgg(firstuser, self.realm, self.factory)
    self._login(user, 'firstuser', 'wrongpass')
    response = self._response(user, 'PRIVMSG')
    self.assertEqual(len(response), 1)
    self.assertEqual(response[0][2], ['firstuser', 'Login failed.  Goodbye.'])