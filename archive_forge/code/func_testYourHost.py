import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def testYourHost(self):
    msg = 'Your host is some.host[blah.blah/6667], running version server-version-3'
    self._serverTestImpl('002', msg, 'yourHost', info=msg)