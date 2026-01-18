import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_no_mismatch():
    with closing_socketpair(socket.AF_UNIX) as pair:
        for sock in pair:
            assert not is_fifo(sock)
            assert not is_mq_wrapper(sock)
            assert not is_socket_inet(sock)
            with skip_enosys():
                assert not is_socket_sockaddr(sock, '127.0.0.1:2000')
            fd = sock.fileno()
            assert not is_fifo(fd)
            assert not is_mq_wrapper(fd)
            assert not is_socket_inet(fd)
            with skip_enosys():
                assert not is_socket_sockaddr(fd, '127.0.0.1:2000')
            assert not _is_fifo(fd)
            assert not _is_mq_wrapper(fd)
            assert not _is_socket_inet(fd)
            with skip_enosys():
                assert not _is_socket_sockaddr(fd, '127.0.0.1:2000')