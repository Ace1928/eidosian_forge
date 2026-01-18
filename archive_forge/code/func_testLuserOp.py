import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def testLuserOp(self):
    args = '34'
    msg = 'flagged staff members'
    self._serverTestImpl('252', msg, 'luserOp', args=args, ops=int(args))