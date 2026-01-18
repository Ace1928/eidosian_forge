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
def process_fd(self, read_fd, write_fd):
    _lib.ares_process_fd(self._channel[0], _ffi.cast('ares_socket_t', read_fd), _ffi.cast('ares_socket_t', write_fd))