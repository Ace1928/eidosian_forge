import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def testGetTopic(self):
    user = self._loggedInUser('useruser')
    group = service.Group('somechannel')
    group.meta['topic'] = 'This is a test topic.'
    group.meta['topic_author'] = 'some_fellow'
    group.meta['topic_date'] = 77777777
    self.successResultOf(self.realm.addGroup(group))
    user.transport.clear()
    user.write('JOIN #somechannel\r\n')
    response = self._response(user)
    self.assertEqual(response[3][0], 'realmname')
    self.assertEqual(response[3][1], '332')
    self.assertEqual(response[3][2], ['useruser', '#somechannel', 'This is a test topic.'])
    self.assertEqual(response[4][1], '333')
    self.assertEqual(response[4][2], ['useruser', '#somechannel', 'some_fellow', '77777777'])
    user.transport.clear()
    user.write('TOPIC #somechannel\r\n')
    response = self._response(user)
    self.assertEqual(response[0][1], '332')
    self.assertEqual(response[0][2], ['useruser', '#somechannel', 'This is a test topic.'])
    self.assertEqual(response[1][1], '333')
    self.assertEqual(response[1][2], ['useruser', '#somechannel', 'some_fellow', '77777777'])