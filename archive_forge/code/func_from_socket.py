from __future__ import annotations
import collections
import contextlib
import functools
import itertools
import os
import socket
import sys
import threading
from debugpy.common import json, log, util
from debugpy.common.util import hide_thread_from_debugger
@classmethod
def from_socket(cls, sock, name=None):
    """Creates a new instance that sends and receives messages over a socket."""
    sock.settimeout(None)
    if name is None:
        name = repr(sock)
    socket_io = sock.makefile('rwb', 0)

    def cleanup():
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        sock.close()
    return cls(socket_io, socket_io, name, cleanup)