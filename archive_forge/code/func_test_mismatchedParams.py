import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_mismatchedParams(self):
    """
        If the number of mode parameters does not match the number of modes
        expecting parameters, L{irc.IRCBadModes} is raised.
        """
    self.assertRaises(irc.IRCBadModes, irc.parseModes, '+k', [], self.paramModes)
    self.assertRaises(irc.IRCBadModes, irc.parseModes, '+kl', ['foo', '10', 'lulz_extra_param'], self.paramModes)