import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_erroneousNick(self):
    """
        Trying to register an illegal nickname results in the default legal
        nickname being set, and trying to change a nickname to an illegal
        nickname results in the old nickname being kept.
        """
    badnick = 'foo'
    self.assertEqual(self.protocol._registered, False)
    self.protocol.register(badnick)
    self.protocol.irc_ERR_ERRONEUSNICKNAME('prefix', ['param'])
    lastLine = self.getLastLine(self.transport)
    self.assertEqual(lastLine, f'NICK {self.protocol.erroneousNickFallback}')
    self.protocol.irc_RPL_WELCOME('prefix', ['param'])
    self.assertEqual(self.protocol._registered, True)
    self.protocol.setNick(self.protocol.erroneousNickFallback)
    self.assertEqual(self.protocol.nickname, self.protocol.erroneousNickFallback)
    oldnick = self.protocol.nickname
    self.protocol.setNick(badnick)
    self.protocol.irc_ERR_ERRONEUSNICKNAME('prefix', ['param'])
    lastLine = self.getLastLine(self.transport)
    self.assertEqual(lastLine, f'NICK {badnick}')
    self.assertEqual(self.protocol.nickname, oldnick)