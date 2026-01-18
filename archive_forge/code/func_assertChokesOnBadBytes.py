from twisted.cred import checkers, portal
from twisted.test import proto_helpers
from twisted.words.protocols import irc
from twisted.words.service import InMemoryWordsRealm, IRCFactory, IRCUser
from twisted.words.test.test_irc import IRCTestCase
def assertChokesOnBadBytes(self, irc_x, error):
    """
        Asserts that IRCUser sends the relevant error code when a given irc_x
        dispatch method is given undecodable bytes.

        @param irc_x: the name of the irc_FOO method to test.
        For example, irc_x = 'PRIVMSG' will check irc_PRIVMSG

        @param error: the error code irc_x should send. For example,
        irc.ERR_NOTONCHANNEL
        """
    getattr(self.ircUser, 'irc_%s' % irc_x)(None, [BADTEXT])
    self.assertEqual(self.ircUser.mockedCodes, [error])