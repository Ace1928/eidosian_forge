import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_getChannelModeParams(self):
    """
        L{IRCClient.getChannelModeParams} uses ISUPPORT information, either
        given by the server or defaults, to determine which channel modes
        require arguments when being added or removed.
        """
    add, remove = map(sorted, self.client.getChannelModeParams())
    self.assertEqual(add, ['b', 'h', 'k', 'l', 'o', 'v'])
    self.assertEqual(remove, ['b', 'h', 'o', 'v'])

    def removeFeature(name):
        name = '-' + name
        msg = 'are available on this server'
        self._serverTestImpl('005', msg, 'isupport', args=name, options=[name])
        self.assertIdentical(self.client.supported.getFeature(name), None)
        self.client.calls = []
    removeFeature('CHANMODES')
    add, remove = map(sorted, self.client.getChannelModeParams())
    self.assertEqual(add, ['h', 'o', 'v'])
    self.assertEqual(remove, ['h', 'o', 'v'])
    removeFeature('PREFIX')
    add, remove = map(sorted, self.client.getChannelModeParams())
    self.assertEqual(add, [])
    self.assertEqual(remove, [])
    self._sendISUPPORT()
    self.assertNotIdentical(self.client.supported.getFeature('PREFIX'), None)