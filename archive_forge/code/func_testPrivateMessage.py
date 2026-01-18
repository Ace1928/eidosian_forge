import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def testPrivateMessage(self):
    user = self._loggedInUser('useruser')
    other = self._loggedInUser('otheruser')
    user.transport.clear()
    other.transport.clear()
    user.write('PRIVMSG otheruser :Hello, monkey.\r\n')
    response = self._response(user)
    event = self._response(other)
    self.assertFalse(response)
    self.assertEqual(len(event), 1)
    self.assertEqual(event[0][0], 'useruser!useruser@realmname')
    self.assertEqual(event[0][1], 'PRIVMSG')
    self.assertEqual(event[0][2], ['otheruser', 'Hello, monkey.'])
    user.write('PRIVMSG nousernamedthis :Hello, monkey.\r\n')
    response = self._response(user)
    self.assertEqual(len(response), 1)
    self.assertEqual(response[0][0], 'realmname')
    self.assertEqual(response[0][1], '401')
    self.assertEqual(response[0][2], ['useruser', 'nousernamedthis', 'No such nick/channel.'])