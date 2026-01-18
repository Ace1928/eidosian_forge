import socket
import struct
from collections import deque
from errno import EAGAIN, EBADF, EINTR, EINVAL, ENOBUFS, ENOSYS, EPERM, EWOULDBLOCK
from functools import wraps
from zope.interface import implementer
from twisted.internet.protocol import DatagramProtocol
from twisted.pair.ethernet import EthernetProtocol
from twisted.pair.ip import IPProtocol
from twisted.pair.rawudp import RawUDPProtocol
from twisted.pair.tuntap import _IFNAMSIZ, _TUNSETIFF, TunnelFlags, _IInputOutputSystem
from twisted.python.compat import nativeString
@wraps(original)
def permissionChecker(self, *args, **kwargs):
    if original.__name__ not in self.permissions:
        raise OSError(EPERM, 'Operation not permitted')
    return original(self, *args, **kwargs)