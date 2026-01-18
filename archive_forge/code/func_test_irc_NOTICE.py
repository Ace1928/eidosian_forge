import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_irc_NOTICE(self):
    """
        L{IRCClient.noticed} is called when a notice is received.
        """
    msg = '%(X)cextended%(X)cdata1%(X)cextended%(X)cdata2%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF}
    self.client.irc_NOTICE(self.user, [self.channel, msg])
    self.assertEqual(self.client.methods, [('noticed', (self.user, '#twisted', 'data1 data2'))])