import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_is_socket_sockaddr():
    with contextlib.closing(socket.socket(socket.AF_INET)) as sock:
        sock.bind(('127.0.0.1', 0))
        addr, port = sock.getsockname()
        port = ':{}'.format(port)
        for listening in (0, 1):
            for arg in (sock, sock.fileno()):
                with skip_enosys():
                    assert is_socket_sockaddr(arg, '127.0.0.1', socket.SOCK_STREAM)
                with skip_enosys():
                    assert is_socket_sockaddr(arg, '127.0.0.1' + port, socket.SOCK_STREAM)
                with skip_enosys():
                    assert is_socket_sockaddr(arg, '127.0.0.1' + port, listening=listening)
                with skip_enosys():
                    assert is_socket_sockaddr(arg, '127.0.0.1' + port, listening=-1)
                with skip_enosys():
                    assert not is_socket_sockaddr(arg, '127.0.0.1' + port, listening=not listening)
                with pytest.raises(ValueError):
                    is_socket_sockaddr(arg, '127.0.0.1', flowinfo=123456)
                with skip_enosys():
                    assert not is_socket_sockaddr(arg, '129.168.11.11:23', socket.SOCK_STREAM)
                with skip_enosys():
                    assert not is_socket_sockaddr(arg, '127.0.0.1', socket.SOCK_DGRAM)
            with pytest.raises(ValueError):
                _is_socket_sockaddr(arg, '127.0.0.1', 0, 123456)
            with skip_enosys():
                assert not _is_socket_sockaddr(arg, '129.168.11.11:23', socket.SOCK_STREAM)
            with skip_enosys():
                assert not _is_socket_sockaddr(arg, '127.0.0.1', socket.SOCK_DGRAM)
            sock.listen(11)