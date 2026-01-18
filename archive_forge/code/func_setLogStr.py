import errno
import socket
import struct
import warnings
from typing import Optional
from zope.interface import implementer
from twisted.internet import address, defer, error, interfaces
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.iocpreactor import abstract, iocpsupport as _iocp
from twisted.internet.iocpreactor.const import (
from twisted.internet.iocpreactor.interfaces import IReadWriteHandle
from twisted.python import failure, log
def setLogStr(self):
    """
        Initialize the C{logstr} attribute to be used by C{logPrefix}.
        """
    logPrefix = self._getLogPrefix(self.protocol)
    self.logstr = '%s (UDP)' % logPrefix