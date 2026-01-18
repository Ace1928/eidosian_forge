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
def test_maxPacket(self):
    """
        An L{SSHChannel} can be configured with a maximum packet size to
        receive.
        """
    channel = self._ourServerOurClientTest(localWindow=11, localMaxPacket=1)

    def cbChannel(channel):
        self.channel = channel
        return channel.conn.sendRequest(channel, b'exec', common.NS(b'secho hello'), 1)
    channel.addCallback(cbChannel)

    def cbExec(ignored):
        return self.channel.onClose
    channel.addCallback(cbExec)

    def cbClosed(ignored):
        self.assertEqual(self.channel.status, 0)
        self.assertEqual(b''.join(self.channel.received), b'hello\r\n')
        self.assertEqual(b''.join(self.channel.receivedExt), b'hello\r\n')
        self.assertEqual(self.channel.localWindowLeft, 11)
        self.assertTrue(self.channel.eofCalled)
    channel.addCallback(cbClosed)
    return channel