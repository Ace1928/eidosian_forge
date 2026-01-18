import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def testWho(self):
    group = service.Group('groupname')
    self.successResultOf(self.realm.addGroup(group))
    users = []
    for nick in ('userone', 'usertwo', 'userthree'):
        u = self._loggedInUser(nick)
        users.append(u)
        users[-1].write('JOIN #groupname\r\n')
    for user in users:
        user.transport.clear()
    users[0].write('WHO #groupname\r\n')
    r = self._response(users[0])
    self.assertFalse(self._response(users[1]))
    self.assertFalse(self._response(users[2]))
    wantusers = ['userone', 'usertwo', 'userthree']
    for prefix, code, stuff in r[:-1]:
        self.assertEqual(prefix, 'realmname')
        self.assertEqual(code, '352')
        myname, group, theirname, theirhost, theirserver, theirnick, flag, extra = stuff
        self.assertEqual(myname, 'userone')
        self.assertEqual(group, '#groupname')
        self.assertTrue(theirname in wantusers)
        self.assertEqual(theirhost, 'realmname')
        self.assertEqual(theirserver, 'realmname')
        wantusers.remove(theirnick)
        self.assertEqual(flag, 'H')
        self.assertEqual(extra, '0 ' + theirnick)
    self.assertFalse(wantusers)
    prefix, code, stuff = r[-1]
    self.assertEqual(prefix, 'realmname')
    self.assertEqual(code, '315')
    myname, channel, extra = stuff
    self.assertEqual(myname, 'userone')
    self.assertEqual(channel, '#groupname')
    self.assertEqual(extra, 'End of /WHO list.')