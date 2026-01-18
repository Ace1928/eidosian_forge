import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_colorFormatting(self):
    """
        Correctly formatted text with colors uses 2 digits to specify
        foreground and (optionally) background.
        """
    self.assertEqual(irc.parseFormattedText('\x0301yay\x03'), A.fg.black['yay'])
    self.assertEqual(irc.parseFormattedText('\x0301,02yay\x03'), A.fg.black[A.bg.blue['yay']])
    self.assertEqual(irc.parseFormattedText('\x0301yay\x0302yipee\x03'), A.fg.black['yay', A.fg.blue['yipee']])