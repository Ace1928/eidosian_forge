from __future__ import absolute_import
import io
import logging
import zlib
from contextlib import contextmanager
from socket import error as SocketError
from socket import timeout as SocketTimeout
from ._collections import HTTPHeaderDict
from .connection import BaseSSLError, HTTPException
from .exceptions import (
from .packages import six
from .util.response import is_fp_closed, is_response_to_head
def drain_conn(self):
    """
        Read and discard any remaining HTTP response data in the response connection.

        Unread data in the HTTPResponse connection blocks the connection from being released back to the pool.
        """
    try:
        self.read()
    except (HTTPError, SocketError, BaseSSLError, HTTPException):
        pass