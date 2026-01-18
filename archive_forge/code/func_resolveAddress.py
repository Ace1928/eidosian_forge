from __future__ import annotations
import os
import socket
import struct
import sys
from typing import Callable, ClassVar, List, Optional, Union
from zope.interface import Interface, implementer
import attr
import typing_extensions
from twisted.internet.interfaces import (
from twisted.logger import ILogObserver, LogEvent, Logger
from twisted.python import deprecate, versions
from twisted.python.compat import lazyByteSlice
from twisted.python.runtime import platformType
from errno import errorcode
from twisted.internet import abstract, address, base, error, fdesc, main
from twisted.internet.error import CannotListenError
from twisted.internet.protocol import Protocol
from twisted.internet.task import deferLater
from twisted.python import failure, log, reflect
from twisted.python.util import untilConcludes
def resolveAddress(self):
    """
        Resolve the name that was passed to this L{_BaseBaseClient}, if
        necessary, and then move on to attempting the connection once an
        address has been determined.  (The connection will be attempted
        immediately within this function if either name resolution can be
        synchronous or the address was an IP address literal.)

        @note: You don't want to call this method from outside, as it won't do
            anything useful; it's just part of the connection bootstrapping
            process.  Also, although this method is on L{_BaseBaseClient} for
            historical reasons, it's not used anywhere except for L{Client}
            itself.

        @return: L{None}
        """
    if self._requiresResolution:
        d = self.reactor.resolve(self.addr[0])
        d.addCallback(lambda n: (n,) + self.addr[1:])
        d.addCallbacks(self._setRealAddress, self.failIfNotConnected)
    else:
        self._setRealAddress(self.addr)