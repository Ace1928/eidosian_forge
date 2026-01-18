from abc import ABCMeta
import copyreg
import functools
import io
import os
import pickle
import socket
import sys
from . import context
def recvfds(sock, size):
    """Receive an array of fds over an AF_UNIX socket."""
    a = array.array('i')
    bytes_size = a.itemsize * size
    msg, ancdata, flags, addr = sock.recvmsg(1, socket.CMSG_SPACE(bytes_size))
    if not msg and (not ancdata):
        raise EOFError
    try:
        if ACKNOWLEDGE:
            sock.send(b'A')
        if len(ancdata) != 1:
            raise RuntimeError('received %d items of ancdata' % len(ancdata))
        cmsg_level, cmsg_type, cmsg_data = ancdata[0]
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            if len(cmsg_data) % a.itemsize != 0:
                raise ValueError
            a.frombytes(cmsg_data)
            if len(a) % 256 != msg[0]:
                raise AssertionError('Len is {0:n} but msg[0] is {1!r}'.format(len(a), msg[0]))
            return list(a)
    except (ValueError, IndexError):
        pass
    raise RuntimeError('Invalid data received')