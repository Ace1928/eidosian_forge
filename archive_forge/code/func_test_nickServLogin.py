import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def test_nickServLogin(self):
    """
        Sending NICK without PASS will prompt the user for their password.
        When the user sends their password to NickServ, it will respond with a
        Greeting.
        """
    firstuser = self.successResultOf(self.realm.lookupUser('firstuser'))
    user = TestCaseUserAgg(firstuser, self.realm, self.factory)
    user.write('NICK firstuser extrainfo\r\n')
    response = self._response(user, 'PRIVMSG')
    self.assertEqual(len(response), 1)
    self.assertEqual(response[0][0], service.NICKSERV)
    self.assertEqual(response[0][1], 'PRIVMSG')
    self.assertEqual(response[0][2], ['firstuser', 'Password?'])
    user.transport.clear()
    user.write('PRIVMSG nickserv firstuser_password\r\n')
    self._assertGreeting(user)