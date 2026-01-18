import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_failToOpenLocalFile(self):
    """
        L{IOError} is raised when failing to open the requested path.
        """
    fp = FilePath(self.mktemp()).child('child-with-no-existing-parent')
    self.assertRaises(IOError, self.makeConnectedDccFileReceive, fp.path)