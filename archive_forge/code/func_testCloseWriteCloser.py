import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
def testCloseWriteCloser(self):
    client = self.client
    f = self.f
    t = client.transport
    t.write(b'hello')
    d = loopUntil(lambda: len(t._tempDataBuffer) == 0)

    def loseWrite(ignored):
        t.loseWriteConnection()
        return loopUntil(lambda: t._writeDisconnected)

    def check(ignored):
        self.assertFalse(client.closed)
        self.assertTrue(client.writeHalfClosed)
        self.assertFalse(client.readHalfClosed)
        return loopUntil(lambda: f.protocol.readHalfClosed)

    def write(ignored):
        w = client.transport.write
        w(b' world')
        w(b'lalala fooled you')
        self.assertEqual(0, len(client.transport._tempDataBuffer))
        self.assertEqual(f.protocol.data, b'hello')
        self.assertFalse(f.protocol.closed)
        self.assertTrue(f.protocol.readHalfClosed)
    return d.addCallback(loseWrite).addCallback(check).addCallback(write)