import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_ctcpQuery_USERINFO(self):
    """
        L{IRCClient.ctcpQuery_USERINFO} calls L{IRCClient.ctcpMakeReply} with
        the correct args.
        """
    self.client.userinfo = 'info'
    self.client.ctcpQuery_USERINFO(self.user, self.channel, 'data')
    self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('USERINFO', 'info')]))])