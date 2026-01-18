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
def read_trailer_lines(self):
    """Read HTTP headers and yield them.

        :yields: CRLF separated lines
        :ytype: bytes

        """
    if not self.closed:
        raise ValueError('Cannot read trailers until the request body has been read.')
    while True:
        line = self.rfile.readline()
        if not line:
            raise ValueError('Illegal end of headers.')
        self.bytes_read += len(line)
        if self.maxlen and self.bytes_read > self.maxlen:
            raise IOError('Request Entity Too Large')
        if line == CRLF:
            break
        if not line.endswith(CRLF):
            raise ValueError('HTTP requires CRLF terminators')
        yield line