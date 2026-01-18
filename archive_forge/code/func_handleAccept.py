from __future__ import annotations
import errno
import socket
import struct
from typing import TYPE_CHECKING, Optional, Union
from zope.interface import classImplements, implementer
from twisted.internet import address, defer, error, interfaces, main
from twisted.internet.abstract import _LogOwner, isIPv6Address
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.interfaces import IProtocol
from twisted.internet.iocpreactor import abstract, iocpsupport as _iocp
from twisted.internet.iocpreactor.const import (
from twisted.internet.iocpreactor.interfaces import IReadWriteHandle
from twisted.internet.protocol import Protocol
from twisted.internet.tcp import (
from twisted.python import failure, log, reflect
def handleAccept(self, rc, evt):
    if self.disconnecting or self.disconnected:
        return False
    if rc:
        log.msg('Could not accept new connection -- %s (%s)' % (errno.errorcode.get(rc, 'unknown error'), rc))
        return False
    else:
        evt.newskt.setsockopt(socket.SOL_SOCKET, SO_UPDATE_ACCEPT_CONTEXT, struct.pack('P', self.socket.fileno()))
        family, lAddr, rAddr = _iocp.get_accept_addrs(evt.newskt.fileno(), evt.buff)
        assert family == self.addressFamily
        if '%' in lAddr[0]:
            scope = int(lAddr[0].split('%')[1])
            lAddr = (lAddr[0], lAddr[1], 0, scope)
        if '%' in rAddr[0]:
            scope = int(rAddr[0].split('%')[1])
            rAddr = (rAddr[0], rAddr[1], 0, scope)
        protocol = self.factory.buildProtocol(self._addressType('TCP', *rAddr))
        if protocol is None:
            evt.newskt.close()
        else:
            s = self.sessionno
            self.sessionno = s + 1
            transport = Server(evt.newskt, protocol, self._addressType('TCP', *rAddr), self._addressType('TCP', *lAddr), s, self.reactor)
            protocol.makeConnection(transport)
        return True