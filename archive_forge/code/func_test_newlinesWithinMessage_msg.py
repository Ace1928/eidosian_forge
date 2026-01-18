import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_newlinesWithinMessage_msg(self):
    """
        An LF within a message causes a new line.
        """
    self.client.lines = []
    self.client.msg('foo', 'bar\nbaz')
    self.assertEqual(self.client.lines, ['PRIVMSG foo :bar', 'PRIVMSG foo :baz'])