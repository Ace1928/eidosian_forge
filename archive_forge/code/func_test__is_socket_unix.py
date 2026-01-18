import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test__is_socket_unix():
    with closing_socketpair(socket.AF_UNIX) as pair:
        for sock in pair:
            fd = sock.fileno()
            assert _is_socket_unix(fd)
            assert not _is_socket_unix(fd, 0, -1, '/no/such/path')
            assert _is_socket_unix(fd, socket.SOCK_STREAM)
            assert not _is_socket_unix(fd, socket.SOCK_DGRAM)