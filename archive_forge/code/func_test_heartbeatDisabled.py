import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_heartbeatDisabled(self):
    """
        If L{irc.IRCClient.heartbeatInterval} is set to L{None} then no
        heartbeat is created.
        """
    self.assertIdentical(self.client._heartbeat, None)
    self.client.heartbeatInterval = None
    self.client.irc_RPL_WELCOME('foo', [])
    self.assertIdentical(self.client._heartbeat, None)