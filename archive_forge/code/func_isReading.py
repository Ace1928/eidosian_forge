import socket
import sys
from typing import Sequence
from zope.interface import classImplements, implementer
from twisted.internet import error, tcp, udp
from twisted.internet.base import ReactorBase
from twisted.internet.interfaces import (
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform, platformType
from ._signals import (
def isReading(self, fd):
    """
        Checks if the file descriptor is currently being observed for read
        readiness.

        @param fd: The file descriptor being checked.
        @type fd: L{twisted.internet.abstract.FileDescriptor}
        @return: C{True} if the file descriptor is being observed for read
            readiness, C{False} otherwise.
        @rtype: C{bool}
        """
    return fd in self._readers