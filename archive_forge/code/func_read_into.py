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
def read_into(self, buf: bytearray, partial: bool=False) -> Awaitable[int]:
    """Asynchronously read a number of bytes.

        ``buf`` must be a writable buffer into which data will be read.

        If ``partial`` is true, the callback is run as soon as any bytes
        have been read.  Otherwise, it is run when the ``buf`` has been
        entirely filled with read data.

        .. versionadded:: 5.0

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.

        """
    future = self._start_read()
    available_bytes = self._read_buffer_size
    n = len(buf)
    if available_bytes >= n:
        buf[:] = memoryview(self._read_buffer)[:n]
        del self._read_buffer[:n]
        self._after_user_read_buffer = self._read_buffer
    elif available_bytes > 0:
        buf[:available_bytes] = memoryview(self._read_buffer)[:]
    self._user_read_buffer = True
    self._read_buffer = buf
    self._read_buffer_size = available_bytes
    self._read_bytes = n
    self._read_partial = partial
    try:
        self._try_inline_read()
    except:
        future.add_done_callback(lambda f: f.exception())
        raise
    return future