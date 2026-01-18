import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_dccChat(self):
    """
        L{irc.IRCClient.dcc_CHAT} invokes L{irc.IRCClient.dccDoChat}.
        """
    self.client.dcc_CHAT(self.user, self.channel, 'foo.txt 127.0.0.1 1025')
    self.assertEqual(self.client.methods, [('dccDoChat', (self.user, self.channel, '127.0.0.1', 1025, ['foo.txt', '127.0.0.1', '1025']))])