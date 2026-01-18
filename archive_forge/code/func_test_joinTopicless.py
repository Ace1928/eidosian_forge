import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def test_joinTopicless(self):
    """
        When a user joins a group without a topic, no topic information is
        sent to that user.
        """
    firstuser = self.successResultOf(self.realm.lookupUser('firstuser'))
    self.successResultOf(self.realm.createGroup('somechannel'))
    user = TestCaseUserAgg(firstuser, self.realm, self.factory)
    self._login(user, 'firstuser')
    user.transport.clear()
    user.write('JOIN #somechannel\r\n')
    response = self._response(user)
    responseCodes = [r[1] for r in response]
    self.assertNotIn('332', responseCodes)
    self.assertNotIn('333', responseCodes)