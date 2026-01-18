from __future__ import print_function
import os
import socket
import signal
import threading
from contextlib import closing, contextmanager
from . import _gi
def ensure_socket_not_inheritable(sock):
    """Ensures that the socket is not inherited by child processes

    Raises:
        EnvironmentError
        NotImplementedError: With Python <3.4 on Windows
    """
    if hasattr(sock, 'set_inheritable'):
        sock.set_inheritable(False)
    else:
        try:
            import fcntl
        except ImportError:
            raise NotImplementedError('Not implemented for older Python on Windows')
        else:
            fd = sock.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFD)
            fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)