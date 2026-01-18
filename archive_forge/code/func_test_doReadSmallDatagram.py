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
def test_doReadSmallDatagram(self):
    """
        L{TuntapPort.doRead} reads a datagram of fewer than
        C{TuntapPort.maxPacketSize} from the port's file descriptor and passes
        it to its protocol's C{datagramReceived} method.
        """
    datagram = b'x' * (self.port.maxPacketSize - 1)
    self.port.startListening()
    tunnel = self.system.getTunnel(self.port)
    tunnel.readBuffer.append(datagram)
    self.port.doRead()
    self.assertEqual([datagram], self.protocol.received)