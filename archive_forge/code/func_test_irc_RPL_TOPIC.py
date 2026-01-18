import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_irc_RPL_TOPIC(self):
    """
        L{IRCClient.topicUpdated} is called when the topic is initially
        reported.
        """
    self.client.irc_RPL_TOPIC(self.user, ['?', self.channel, 'new topic is new'])
    self.assertEqual(self.client.methods, [('topicUpdated', ('Wolf', self.channel, 'new topic is new'))])