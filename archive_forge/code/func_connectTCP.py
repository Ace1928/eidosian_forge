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
def connectTCP(self, host, port, factory, timeout=30, bindAddress=None):
    """
        @see: twisted.internet.interfaces.IReactorTCP.connectTCP
        """
    c = tcp.Connector(host, port, factory, timeout, bindAddress, self)
    c.connect()
    return c