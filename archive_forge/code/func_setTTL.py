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
def setTTL(self, ttl):
    ttl = struct.pack('B', ttl)
    self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)