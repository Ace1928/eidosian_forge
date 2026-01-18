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
def cbPty(ignored):
    session = self.realm.avatar.conn.channels[0].session
    self.assertIs(session.avatar, self.realm.avatar)
    self.assertEqual(session._terminalType, b'conch-test-term')
    self.assertEqual(session._windowSize, (24, 80, 0, 0))
    self.assertTrue(session.ptyReq)
    channel = self.channel
    return channel.conn.sendRequest(channel, b'shell', b'', 1)