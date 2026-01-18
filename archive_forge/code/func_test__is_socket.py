import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test__is_socket():
    with closing_socketpair(socket.AF_UNIX) as pair:
        for sock in pair:
            fd = sock.fileno()
            assert _is_socket(fd)
            assert _is_socket(fd, socket.AF_UNIX)
            assert not _is_socket(fd, socket.AF_INET)
            assert _is_socket(fd, socket.AF_UNIX, socket.SOCK_STREAM)
            assert not _is_socket(fd, socket.AF_INET, socket.SOCK_DGRAM)
            assert _is_socket(fd)
            assert _is_socket(fd, socket.AF_UNIX)
            assert not _is_socket(fd, socket.AF_INET)
            assert _is_socket(fd, socket.AF_UNIX, socket.SOCK_STREAM)
            assert not _is_socket(fd, socket.AF_INET, socket.SOCK_DGRAM)