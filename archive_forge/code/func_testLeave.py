import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def testLeave(self):
    user = self._loggedInUser('useruser')
    self.successResultOf(self.realm.createGroup('somechannel'))
    user.write('JOIN #somechannel\r\n')
    user.transport.clear()
    other = self._loggedInUser('otheruser')
    other.write('JOIN #somechannel\r\n')
    user.transport.clear()
    other.transport.clear()
    user.write('PART #somechannel\r\n')
    response = self._response(user)
    event = self._response(other)
    self.assertEqual(len(response), 1)
    self.assertEqual(response[0][0], 'useruser!useruser@realmname')
    self.assertEqual(response[0][1], 'PART')
    self.assertEqual(response[0][2], ['#somechannel', 'leaving'])
    self.assertEqual(response, event)
    user.write('JOIN #somechannel\r\n')
    user.transport.clear()
    other.transport.clear()
    user.write('PART #somechannel :goodbye stupidheads\r\n')
    response = self._response(user)
    event = self._response(other)
    self.assertEqual(len(response), 1)
    self.assertEqual(response[0][0], 'useruser!useruser@realmname')
    self.assertEqual(response[0][1], 'PART')
    self.assertEqual(response[0][2], ['#somechannel', 'goodbye stupidheads'])
    self.assertEqual(response, event)
    user.write(b'JOIN #somechannel\r\n')
    user.transport.clear()
    other.transport.clear()
    user.write(b'PART #somechannel :goodbye stupidheads1\r\n')
    response = self._response(user)
    event = self._response(other)
    self.assertEqual(len(response), 1)
    self.assertEqual(response[0][0], 'useruser!useruser@realmname')
    self.assertEqual(response[0][1], 'PART')
    self.assertEqual(response[0][2], ['#somechannel', 'goodbye stupidheads1'])
    self.assertEqual(response, event)