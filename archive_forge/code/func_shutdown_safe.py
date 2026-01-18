import errno
import os
import socket
import sys
import time
import warnings
import eventlet
from eventlet.hubs import trampoline, notify_opened, IOClosed
from eventlet.support import get_errno
def shutdown_safe(sock):
    """Shuts down the socket. This is a convenience method for
    code that wants to gracefully handle regular sockets, SSL.Connection
    sockets from PyOpenSSL and ssl.SSLSocket objects from Python 2.7 interchangeably.
    Both types of ssl socket require a shutdown() before close,
    but they have different arity on their shutdown method.

    Regular sockets don't need a shutdown before close, but it doesn't hurt.
    """
    try:
        try:
            return sock.shutdown(socket.SHUT_RDWR)
        except TypeError:
            return sock.shutdown()
    except OSError as e:
        if get_errno(e) not in (errno.ENOTCONN, errno.EBADF, errno.ENOTSOCK):
            raise