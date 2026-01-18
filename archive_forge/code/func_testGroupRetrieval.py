import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def testGroupRetrieval(self):
    realm = service.InMemoryWordsRealm('realmname')
    group = self.successResultOf(realm.createGroup('testgroup'))
    retrieved = self.successResultOf(realm.getGroup('testgroup'))
    self.assertIdentical(group, retrieved)
    self.failureResultOf(realm.getGroup('nosuchgroup')).trap(ewords.NoSuchGroup)