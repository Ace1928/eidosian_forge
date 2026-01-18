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
def test_errorChannel(self):
    """
        Bytes sent over the extended channel for stderr data are delivered to
        the channel's C{extReceived} method.
        """
    channel = self._ourServerOurClientTest(localWindow=4, localMaxPacket=5)

    def cbChannel(channel):
        self.channel = channel
        return channel.conn.sendRequest(channel, b'exec', common.NS(b'eecho hello'), 1)
    channel.addCallback(cbChannel)

    def cbExec(ignored):
        return defer.gatherResults([self.channel.onClose, self.realm.avatar._testSession.onClose])
    channel.addCallback(cbExec)

    def cbClosed(ignored):
        self.assertEqual(self.channel.received, [])
        self.assertEqual(b''.join(self.channel.receivedExt), b'hello\r\n')
        self.assertEqual(self.channel.status, 0)
        self.assertTrue(self.channel.eofCalled)
        self.assertEqual(self.channel.localWindowLeft, 4)
        self.assertEqual(self.channel.localWindowLeft, self.realm.avatar._testSession.remoteWindowLeftAtClose)
    channel.addCallback(cbClosed)
    return channel