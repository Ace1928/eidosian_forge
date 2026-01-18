import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_noModeParameters(self):
    """
        No parameters are passed to L{IRCClient.modeChanged} for modes that
        don't take any parameters.
        """
    self._sendModeChange('-s')
    self._checkModeChange([(False, 's', (None,))])
    self._sendModeChange('+n')
    self._checkModeChange([(True, 'n', (None,))])