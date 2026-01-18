import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def assertAssembledEqually(self, text, expectedFormatted):
    """
        Assert that C{text} is parsed and assembled to the same value as what
        C{expectedFormatted} is assembled to. This provides a way to ignore
        meaningless differences in the formatting structure that would be
        difficult to detect without rendering the structures.
        """
    formatted = irc.parseFormattedText(text)
    self.assertAssemblesTo(formatted, expectedFormatted)