import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_buildProtocol(self):
    """
        An instance of the L{irc.DccChat} protocol is returned, which has the
        factory property set to the factory which created it.
        """
    queryData = ('fromUser', None, None)
    factory = irc.DccChatFactory(None, queryData)
    protocol = factory.buildProtocol('127.0.0.1')
    self.assertIsInstance(protocol, irc.DccChat)
    self.assertEqual(protocol.factory, factory)