import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_mixedModes(self):
    """
        Mixing adding and removing modes that do and don't take parameters
        invokes L{IRCClient.modeChanged} with mode characters and parameters
        that match up.
        """
    self._sendModeChange('+osv', 'a_user another_user')
    self._checkModeChange([(True, 'osv', ('a_user', None, 'another_user'))])
    self._sendModeChange('+v-os', 'a_user another_user')
    self._checkModeChange([(True, 'v', ('a_user',)), (False, 'os', ('another_user', None))])