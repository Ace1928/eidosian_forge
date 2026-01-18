import socket
import sys
import warnings
from typing import Tuple, Type
from zope.interface import implementer
from twisted.internet import base, error, interfaces, main
from twisted.internet._dumbwin32proc import Process
from twisted.internet.iocpreactor import iocpsupport as _iocp, tcp, udp
from twisted.internet.iocpreactor.const import WAIT_TIMEOUT
from twisted.internet.win32eventreactor import _ThreadedWin32EventsMixin
from twisted.python import failure, log
def listenMulticast(self, port, protocol, interface='', maxPacketSize=8192, listenMultiple=False):
    """
        Connects a given DatagramProtocol to the given numeric UDP port.

        EXPERIMENTAL.

        @returns: object conforming to IListeningPort.
        """
    p = udp.MulticastPort(port, protocol, interface, maxPacketSize, self, listenMultiple)
    p.startListening()
    return p