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
def handleRead(self, rc, data, evt):
    if rc in (errno.WSAECONNREFUSED, errno.WSAECONNRESET, ERROR_CONNECTION_REFUSED, ERROR_PORT_UNREACHABLE):
        if self._connectedAddr:
            self.protocol.connectionRefused()
    elif rc:
        log.msg('error in recvfrom -- %s (%s)' % (errno.errorcode.get(rc, 'unknown error'), rc))
    else:
        try:
            self.protocol.datagramReceived(bytes(evt.buff[:data]), _iocp.makesockaddr(evt.addr_buff))
        except BaseException:
            log.err()