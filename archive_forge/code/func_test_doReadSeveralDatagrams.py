import os
import socket
import struct
from collections import deque
from errno import EAGAIN, EBADF, EINVAL, ENODEV, ENOENT, EPERM, EWOULDBLOCK
from itertools import cycle
from random import randrange
from signal import SIGINT
from typing import Optional
from twisted.python.reflect import ObjectNotFound, namedAny
from zope.interface import Interface, implementer
from zope.interface.verify import verifyObject
from twisted.internet.error import CannotListenError
from twisted.internet.interfaces import IAddress, IListeningPort, IReactorFDSet
from twisted.internet.protocol import (
from twisted.internet.task import Clock
from twisted.pair.ethernet import EthernetProtocol
from twisted.pair.ip import IPProtocol
from twisted.pair.raw import IRawPacketProtocol
from twisted.pair.rawudp import RawUDPProtocol
from twisted.python.compat import iterbytes
from twisted.python.log import addObserver, removeObserver, textFromEventDict
from twisted.python.reflect import fullyQualifiedName
from twisted.trial.unittest import SkipTest, SynchronousTestCase
def test_doReadSeveralDatagrams(self):
    """
        L{TuntapPort.doRead} reads several datagrams, of up to
        C{TuntapPort.maxThroughput} bytes total, before returning.
        """
    values = cycle(iterbytes(b'abcdefghijklmnopqrstuvwxyz'))
    total = 0
    datagrams = []
    while total < self.port.maxThroughput:
        datagrams.append(next(values) * self.port.maxPacketSize)
        total += self.port.maxPacketSize
    self.port.startListening()
    tunnel = self.system.getTunnel(self.port)
    tunnel.readBuffer.extend(datagrams)
    tunnel.readBuffer.append(b'excessive datagram, not to be read')
    self.port.doRead()
    self.assertEqual(datagrams, self.protocol.received)