import errno
import os
import socket
import sys
import time
import warnings
import eventlet
from eventlet.hubs import trampoline, notify_opened, IOClosed
from eventlet.support import get_errno
def set_nonblocking(fd):
    """
    Sets the descriptor to be nonblocking.  Works on many file-like
    objects as well as sockets.  Only sockets can be nonblocking on
    Windows, however.
    """
    try:
        setblocking = fd.setblocking
    except AttributeError:
        try:
            import fcntl
        except ImportError:
            raise NotImplementedError("set_nonblocking() on a file object with no setblocking() method (Windows pipes don't support non-blocking I/O)")
        fileno = fd.fileno()
        orig_flags = fcntl.fcntl(fileno, fcntl.F_GETFL)
        new_flags = orig_flags | os.O_NONBLOCK
        if new_flags != orig_flags:
            fcntl.fcntl(fileno, fcntl.F_SETFL, new_flags)
    else:
        setblocking(0)