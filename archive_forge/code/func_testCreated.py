import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def testCreated(self):
    msg = 'This server was cobbled together Fri Aug 13 18:00:25 UTC 2004'
    self._serverTestImpl('003', msg, 'created', when=msg)