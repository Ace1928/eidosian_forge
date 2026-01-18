import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_dccSend(self):
    """
        L{irc.IRCClient.dcc_SEND} invokes L{irc.IRCClient.dccDoSend}.
        """
    self.client.dcc_SEND(self.user, self.channel, 'foo.txt 127.0.0.1 1025')
    self.assertEqual(self.client.methods, [('dccDoSend', (self.user, '127.0.0.1', 1025, 'foo.txt', -1, ['foo.txt', '127.0.0.1', '1025']))])