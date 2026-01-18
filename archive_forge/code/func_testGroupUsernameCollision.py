import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def testGroupUsernameCollision(self):
    """
        Try creating a group with the same name as an existing user and
        assert that it succeeds, since users and groups should not be in the
        same namespace and collisions should be impossible.
        """
    realm = service.InMemoryWordsRealm('realmname')
    self.successResultOf(realm.createUser('test'))
    self.successResultOf(realm.createGroup('test'))