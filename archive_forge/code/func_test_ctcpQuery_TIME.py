import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_ctcpQuery_TIME(self):
    """
        L{IRCClient.ctcpQuery_TIME} calls L{IRCClient.ctcpMakeReply} with the
        correct args.
        """
    self.client.ctcpQuery_TIME(self.user, self.channel, 'data')
    self.assertEqual(self.client.methods[0][1][0], 'Wolf')