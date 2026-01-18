import os
import io
import re
import email.utils
import socket
import sys
import time
import traceback as traceback_
import logging
import platform
import queue
import contextlib
import threading
import urllib.parse
from functools import lru_cache
from . import connections, errors, __version__
from ._compat import bton
from ._compat import IS_PPC
from .workers import threadpool
from .makefile import MakeFile, StreamWriter
@classmethod
def prepare_socket(cls, bind_addr, family, type, proto, nodelay, ssl_adapter, reuse_port=False):
    """Create and prepare the socket object."""
    sock = socket.socket(family, type, proto)
    connections.prevent_socket_inheritance(sock)
    host, port = bind_addr[:2]
    IS_EPHEMERAL_PORT = port == 0
    if reuse_port:
        cls._make_socket_reusable(socket_=sock, bind_addr=bind_addr)
    if not (IS_WINDOWS or IS_EPHEMERAL_PORT):
        'Enable SO_REUSEADDR for the current socket.\n\n            Skip for Windows (has different semantics)\n            or ephemeral ports (can steal ports from others).\n\n            Refs:\n            * https://msdn.microsoft.com/en-us/library/ms740621(v=vs.85).aspx\n            * https://github.com/cherrypy/cheroot/issues/114\n            * https://gavv.github.io/blog/ephemeral-port-reuse/\n            '
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if nodelay and (not isinstance(bind_addr, (str, bytes))):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if ssl_adapter is not None:
        sock = ssl_adapter.bind(sock)
    listening_ipv6 = hasattr(socket, 'AF_INET6') and family == socket.AF_INET6 and (host in ('::', '::0', '::0.0.0.0'))
    if listening_ipv6:
        try:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except (AttributeError, socket.error):
            pass
    return sock