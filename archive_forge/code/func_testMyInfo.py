import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def testMyInfo(self):
    msg = 'server.host server-version abcDEF bcdEHI'
    self._serverTestImpl('004', msg, 'myInfo', servername='server.host', version='server-version', umodes='abcDEF', cmodes='bcdEHI')