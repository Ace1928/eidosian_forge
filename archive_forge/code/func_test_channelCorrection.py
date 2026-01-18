import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_channelCorrection(self):
    """
        If the channel name passed to L{IRCClient.invite} does not begin with a
        channel prefix character, one is prepended to it.
        """
    self.client.invite('foo', 'bar')
    self.assertEqual(self.client.lines, ['INVITE foo #bar'])