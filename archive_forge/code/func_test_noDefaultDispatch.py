import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_noDefaultDispatch(self):
    """
        The fallback handler is invoked for unrecognized CTCP messages.
        """

    def unknownQuery(user, channel, tag, data):
        self.calledWith = (user, channel, tag, data)
        self.called += 1
    self.called = 0
    self.patch(self.client, 'ctcpUnknownQuery', unknownQuery)
    self.client.irc_PRIVMSG('foo!bar@baz.quux', ['#chan', '{X}NOTREAL{X}'.format(X=irc.X_DELIM)])
    self.assertEqualBufferValue(self.file.getvalue(), '')
    self.assertEqual(self.calledWith, ('foo!bar@baz.quux', '#chan', 'NOTREAL', None))
    self.assertEqual(self.called, 1)
    self.client.irc_PRIVMSG('foo!bar@baz.quux', ['#chan', '{X}NOTREAL{X}foo{X}NOTREAL{X}'.format(X=irc.X_DELIM)])
    self.assertEqual(self.called, 2)