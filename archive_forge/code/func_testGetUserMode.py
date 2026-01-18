import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def testGetUserMode(self):
    user = self._loggedInUser('useruser')
    user.transport.clear()
    user.write('MODE useruser\r\n')
    response = self._response(user)
    self.assertEqual(len(response), 1)
    self.assertEqual(response[0][0], 'realmname')
    self.assertEqual(response[0][1], '221')
    self.assertEqual(response[0][2], ['useruser', '+'])