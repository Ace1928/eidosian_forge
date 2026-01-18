import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def testLuserMe(self):
    msg = 'I have 1937 clients and 0 servers'
    self._serverTestImpl('255', msg, 'luserMe', info=msg)