import errno
import os
import socket
import ssl
from tornado import gen
from tornado.log import app_log
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream, SSLIOStream
from tornado.netutil import (
from tornado import process
from tornado.util import errno_from_exception
import typing
from typing import Union, Dict, Any, Iterable, Optional, Awaitable
def add_sockets(self, sockets: Iterable[socket.socket]) -> None:
    """Makes this server start accepting connections on the given sockets.

        The ``sockets`` parameter is a list of socket objects such as
        those returned by `~tornado.netutil.bind_sockets`.
        `add_sockets` is typically used in combination with that
        method and `tornado.process.fork_processes` to provide greater
        control over the initialization of a multi-process server.
        """
    for sock in sockets:
        self._sockets[sock.fileno()] = sock
        self._handlers[sock.fileno()] = add_accept_handler(sock, self._handle_connection)