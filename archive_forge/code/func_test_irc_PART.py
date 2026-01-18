import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_irc_PART(self):
    """
        L{IRCClient.left} is called when I part the channel;
        L{IRCClient.userLeft} is called when someone else parts.
        """
    self.client.irc_PART(self.user, [self.channel])
    self.client.irc_PART('Svadilfari!~svadi@yok.utu.fi', ['#python'])
    self.assertEqual(self.client.methods, [('left', (self.channel,)), ('userLeft', ('Svadilfari', '#python'))])