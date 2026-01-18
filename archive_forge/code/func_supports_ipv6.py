from __future__ import division
from __future__ import print_function
import collections
import contextlib
import errno
import functools
import os
import socket
import stat
import sys
import threading
import warnings
from collections import namedtuple
from socket import AF_INET
from socket import SOCK_DGRAM
from socket import SOCK_STREAM
@memoize
def supports_ipv6():
    """Return True if IPv6 is supported on this platform."""
    if not socket.has_ipv6 or AF_INET6 is None:
        return False
    try:
        sock = socket.socket(AF_INET6, socket.SOCK_STREAM)
        with contextlib.closing(sock):
            sock.bind(('::1', 0))
        return True
    except socket.error:
        return False