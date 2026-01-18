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
def set_local_ip(self, ip):
    addr4 = _ffi.new('struct in_addr*')
    addr6 = _ffi.new('struct ares_in6_addr*')
    if _lib.ares_inet_pton(socket.AF_INET, ascii_bytes(ip), addr4) == 1:
        _lib.ares_set_local_ip4(self._channel[0], socket.ntohl(addr4.s_addr))
    elif _lib.ares_inet_pton(socket.AF_INET6, ascii_bytes(ip), addr6) == 1:
        _lib.ares_set_local_ip6(self._channel[0], addr6)
    else:
        raise ValueError('invalid IP address')