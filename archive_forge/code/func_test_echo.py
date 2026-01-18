import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
def test_echo(self):
    """
        Normal standard out bytes are sent to the channel's C{dataReceived}
        method.
        """
    channel = self._ourServerOurClientTest(localWindow=4, localMaxPacket=5)

    def cbChannel(channel):
        self.channel = channel
        return channel.conn.sendRequest(channel, b'exec', common.NS(b'echo hello'), 1)
    channel.addCallback(cbChannel)

    def cbEcho(ignored):
        return defer.gatherResults([self.channel.onClose, self.realm.avatar._testSession.onClose])
    channel.addCallback(cbEcho)

    def cbClosed(ignored):
        self.assertEqual(self.channel.status, 0)
        self.assertEqual(b''.join(self.channel.received), b'hello\r\n')
        self.assertEqual(self.channel.localWindowLeft, 4)
        self.assertTrue(self.channel.eofCalled)
        self.assertEqual(self.channel.localWindowLeft, self.realm.avatar._testSession.remoteWindowLeftAtClose)
    channel.addCallback(cbClosed)
    return channel