import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_dccAccept(self):
    """
        L{irc.IRCClient.dcc_ACCEPT} invokes L{irc.IRCClient.dccDoAcceptResume}.
        """
    self.client.dcc_ACCEPT(self.user, self.channel, 'foo.txt 1025 2')
    self.assertEqual(self.client.methods, [('dccDoAcceptResume', (self.user, 'foo.txt', 1025, 2))])