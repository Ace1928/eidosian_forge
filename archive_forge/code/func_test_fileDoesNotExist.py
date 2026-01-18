import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_fileDoesNotExist(self):
    """
        If the file does not already exist, then L{DccFileReceive} will
        create one and write the data to it.
        """
    fp = FilePath(self.mktemp())
    protocol = self.makeConnectedDccFileReceive(fp.path)
    self.allDataReceivedForProtocol(protocol, b'I <3 Twisted')
    self.assertEqual(fp.getContent(), b'I <3 Twisted')