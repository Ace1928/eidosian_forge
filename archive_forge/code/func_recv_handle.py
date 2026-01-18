from abc import ABCMeta
import copyreg
import functools
import io
import os
import pickle
import socket
import sys
from . import context
def recv_handle(conn):
    """Receive a handle over a local connection."""
    with socket.fromfd(conn.fileno(), socket.AF_UNIX, socket.SOCK_STREAM) as s:
        return recvfds(s, 1)[0]