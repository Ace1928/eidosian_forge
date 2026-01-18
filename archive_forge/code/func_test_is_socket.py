import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_is_socket():
    with closing_socketpair(socket.AF_UNIX) as pair:
        for sock in pair:
            for arg in (sock, sock.fileno()):
                assert is_socket(arg)
                assert is_socket(arg, socket.AF_UNIX)
                assert not is_socket(arg, socket.AF_INET)
                assert is_socket(arg, socket.AF_UNIX, socket.SOCK_STREAM)
                assert not is_socket(arg, socket.AF_INET, socket.SOCK_DGRAM)
                with skip_enosys():
                    assert not is_socket_sockaddr(arg, '8.8.8.8:2000', socket.SOCK_DGRAM, 0, 0)
            assert _is_socket(arg)
            assert _is_socket(arg, socket.AF_UNIX)
            assert not _is_socket(arg, socket.AF_INET)
            assert _is_socket(arg, socket.AF_UNIX, socket.SOCK_STREAM)
            assert not _is_socket(arg, socket.AF_INET, socket.SOCK_DGRAM)
            with skip_enosys():
                assert not _is_socket_sockaddr(arg, '8.8.8.8:2000', socket.SOCK_DGRAM, 0, 0)