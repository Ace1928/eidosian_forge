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
@servers.setter
def servers(self, servers):
    c = _ffi.new('struct ares_addr_node[%d]' % len(servers))
    for i, server in enumerate(servers):
        if _lib.ares_inet_pton(socket.AF_INET, ascii_bytes(server), _ffi.addressof(c[i].addr.addr4)) == 1:
            c[i].family = socket.AF_INET
        elif _lib.ares_inet_pton(socket.AF_INET6, ascii_bytes(server), _ffi.addressof(c[i].addr.addr6)) == 1:
            c[i].family = socket.AF_INET6
        else:
            raise ValueError('invalid IP address')
        if i > 0:
            c[i - 1].next = _ffi.addressof(c[i])
    r = _lib.ares_set_servers(self._channel[0], c)
    if r != _lib.ARES_SUCCESS:
        raise AresError(r, errno.strerror(r))