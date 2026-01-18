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
def connectSSL(self, host, port, factory, contextFactory, timeout=30, bindAddress=None):
    """
            Non-implementation of L{IReactorSSL.connectSSL}.  Some dependency
            is not satisfied.  This implementation always raises
            L{NotImplementedError}.
            """
    raise NotImplementedError('pyOpenSSL 0.10 or newer is required for SSL support in iocpreactor. It is missing, so the reactor does not support SSL APIs.')