import os
import io
import re
import email.utils
import socket
import sys
import time
import traceback as traceback_
import logging
import platform
import queue
import contextlib
import threading
import urllib.parse
from functools import lru_cache
from . import connections, errors, __version__
from ._compat import bton
from ._compat import IS_PPC
from .workers import threadpool
from .makefile import MakeFile, StreamWriter
def simple_response(self, status, msg=''):
    """Write a simple response back to the client."""
    status = str(status)
    proto_status = '%s %s\r\n' % (self.server.protocol, status)
    content_length = 'Content-Length: %s\r\n' % len(msg)
    content_type = 'Content-Type: text/plain\r\n'
    buf = [proto_status.encode('ISO-8859-1'), content_length.encode('ISO-8859-1'), content_type.encode('ISO-8859-1')]
    if status[:3] in ('413', '414'):
        self.close_connection = True
        if self.response_protocol == 'HTTP/1.1':
            buf.append(b'Connection: close\r\n')
        else:
            status = '400 Bad Request'
    buf.append(CRLF)
    if msg:
        if isinstance(msg, str):
            msg = msg.encode('ISO-8859-1')
        buf.append(msg)
    try:
        self.conn.wfile.write(EMPTY.join(buf))
    except socket.error as ex:
        if ex.args[0] not in errors.socket_errors_to_ignore:
            raise