import errno
import os
import socket
import sys
import time
import warnings
import eventlet
from eventlet.hubs import trampoline, notify_opened, IOClosed
from eventlet.support import get_errno
def socket_accept(descriptor):
    """
    Attempts to accept() on the descriptor, returns a client,address tuple
    if it succeeds; returns None if it needs to trampoline, and raises
    any exceptions.
    """
    try:
        return descriptor.accept()
    except OSError as e:
        if get_errno(e) == errno.EWOULDBLOCK:
            return None
        raise