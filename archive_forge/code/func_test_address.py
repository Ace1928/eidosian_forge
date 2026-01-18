import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_address(self):
    """
        L{irc.dccDescribe} supports long IP addresses.
        """
    result = irc.dccDescribe('CHAT arg 3232235522 6666')
    self.assertEqual(result, 'CHAT for host 192.168.0.2, port 6666')