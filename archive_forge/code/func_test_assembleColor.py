import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_assembleColor(self):
    """
        A I{foreground} and I{background} color string assembles to a string
        prefixed with the I{off} and I{color} (followed by the relevant
        foreground color, I{,} and the relevant background color code) control
        codes.
        """
    self.assertEqual(irc.assembleFormattedText(A.fg.red[A.bg.blue['hello']]), '\x0f\x0305,02hello')