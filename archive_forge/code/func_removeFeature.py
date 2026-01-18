import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def removeFeature(name):
    name = '-' + name
    msg = 'are available on this server'
    self._serverTestImpl('005', msg, 'isupport', args=name, options=[name])
    self.assertIdentical(self.client.supported.getFeature(name), None)
    self.client.calls = []