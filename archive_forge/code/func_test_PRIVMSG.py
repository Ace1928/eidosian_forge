from twisted.cred import checkers, portal
from twisted.test import proto_helpers
from twisted.words.protocols import irc
from twisted.words.service import InMemoryWordsRealm, IRCFactory, IRCUser
from twisted.words.test.test_irc import IRCTestCase
def test_PRIVMSG(self):
    """
        Tests that irc_PRIVMSG sends ERR_NOSUCHNICK if the target name can't
        be decoded.
        """
    self.assertChokesOnBadBytes('PRIVMSG', irc.ERR_NOSUCHNICK)