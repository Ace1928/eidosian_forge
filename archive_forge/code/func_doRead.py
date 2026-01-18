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
def doRead(self):
    evt = _iocp.Event(self.cbRead, self)
    evt.buff = buff = self._readBuffers[0]
    evt.addr_buff = addr_buff = self.addressBuffer
    evt.addr_len_buff = addr_len_buff = self.addressLengthBuffer
    rc, data = _iocp.recvfrom(self.getFileHandle(), buff, addr_buff, addr_len_buff, evt)
    if rc and rc != ERROR_IO_PENDING:
        self.reactor.callLater(0, self.cbRead, rc, data, evt)