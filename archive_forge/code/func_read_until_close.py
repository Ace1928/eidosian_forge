import asyncio
import collections
import errno
import io
import numbers
import os
import socket
import ssl
import sys
import re
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import ioloop
from tornado.log import gen_log
from tornado.netutil import ssl_wrap_socket, _client_ssl_defaults, _server_ssl_defaults
from tornado.util import errno_from_exception
import typing
from typing import (
from types import TracebackType
def read_until_close(self) -> Awaitable[bytes]:
    """Asynchronously reads all data from the socket until it is closed.

        This will buffer all available data until ``max_buffer_size``
        is reached. If flow control or cancellation are desired, use a
        loop with `read_bytes(partial=True) <.read_bytes>` instead.

        .. versionchanged:: 4.0
            The callback argument is now optional and a `.Future` will
            be returned if it is omitted.

        .. versionchanged:: 6.0

           The ``callback`` and ``streaming_callback`` arguments have
           been removed. Use the returned `.Future` (and `read_bytes`
           with ``partial=True`` for ``streaming_callback``) instead.

        """
    future = self._start_read()
    if self.closed():
        self._finish_read(self._read_buffer_size)
        return future
    self._read_until_close = True
    try:
        self._try_inline_read()
    except:
        future.add_done_callback(lambda f: f.exception())
        raise
    return future