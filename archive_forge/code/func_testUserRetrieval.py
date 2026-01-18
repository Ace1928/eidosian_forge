import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def testUserRetrieval(self):
    realm = service.InMemoryWordsRealm('realmname')
    user = self.successResultOf(realm.createUser('testuser'))
    retrieved = self.successResultOf(realm.getUser('testuser'))
    self.assertIdentical(user, retrieved)
    lookedUp = self.successResultOf(realm.lookupUser('testuser'))
    self.assertIdentical(retrieved, lookedUp)
    self.failureResultOf(realm.lookupUser('nosuchuser')).trap(ewords.NoSuchUser)