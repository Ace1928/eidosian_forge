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
def registerSpecialDevice(self, name, cls):
    """
        Specify a class which will be used to handle I/O to a device of a
        particular name.

        @param name: The filesystem path name of the device.
        @type name: L{bytes}

        @param cls: A class (like L{Tunnel}) to instantiated whenever this
            device is opened.
        """
    self._devices[name] = cls