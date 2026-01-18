from ._cares import ffi as _ffi, lib as _lib
import _cffi_backend  # hint for bundler tools
from . import errno
from .utils import ascii_bytes, maybe_str, parse_name
from ._version import __version__
import collections.abc
import socket
import math
import functools
import sys
def getsock(self):
    rfds = []
    wfds = []
    socks = _ffi.new('ares_socket_t [%d]' % _lib.ARES_GETSOCK_MAXNUM)
    bitmask = _lib.ares_getsock(self._channel[0], socks, _lib.ARES_GETSOCK_MAXNUM)
    for i in range(_lib.ARES_GETSOCK_MAXNUM):
        if _lib.ARES_GETSOCK_READABLE(bitmask, i):
            rfds.append(socks[i])
        if _lib.ARES_GETSOCK_WRITABLE(bitmask, i):
            wfds.append(socks[i])
    return (rfds, wfds)