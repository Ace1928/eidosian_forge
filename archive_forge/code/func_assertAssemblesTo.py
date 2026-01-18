import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def assertAssemblesTo(self, formatted, expectedFormatted):
    """
        Assert that C{formatted} and C{expectedFormatted} assemble to the same
        value.
        """
    text = irc.assembleFormattedText(formatted)
    expectedText = irc.assembleFormattedText(expectedFormatted)
    self.assertEqual(irc.assembleFormattedText(formatted), expectedText, '%r (%r) is not equivalent to %r (%r)' % (text, formatted, expectedText, expectedFormatted))