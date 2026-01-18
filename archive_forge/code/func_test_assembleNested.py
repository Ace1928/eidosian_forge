import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_assembleNested(self):
    """
        Nested attributes retain the attributes of their parents.
        """
    self.assertEqual(irc.assembleFormattedText(A.bold['hello', A.underline[' world']]), '\x0f\x02hello\x0f\x02\x1f world')
    self.assertEqual(irc.assembleFormattedText(A.normal[A.fg.red[A.bg.green['hello'], ' world'], A.reverseVideo[' yay']]), '\x0f\x0305,03hello\x0f\x0305 world\x0f\x16 yay')