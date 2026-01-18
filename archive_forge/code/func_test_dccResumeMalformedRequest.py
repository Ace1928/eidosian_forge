import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_dccResumeMalformedRequest(self):
    """
        L{irc.IRCClient.dcc_RESUME} raises L{irc.IRCBadMessage} when it is
        passed a malformed query string.
        """
    result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_RESUME, self.user, self.channel, 'foo')
    self.assertEqual(str(result), "malformed DCC SEND RESUME request: ['foo']")