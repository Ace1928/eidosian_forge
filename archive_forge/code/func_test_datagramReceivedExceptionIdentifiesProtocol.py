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
def test_datagramReceivedExceptionIdentifiesProtocol(self):
    """
        The exception raised by C{datagramReceived} is logged with a message
        identifying the offending protocol.
        """
    messages = []
    addObserver(messages.append)
    self.addCleanup(removeObserver, messages.append)
    self._datagramReceivedException()
    error = next((m for m in messages if m['isError']))
    message = textFromEventDict(error)
    self.assertEqual('Unhandled exception from %s.datagramReceived' % (fullyQualifiedName(self.protocol.__class__),), message.splitlines()[0])