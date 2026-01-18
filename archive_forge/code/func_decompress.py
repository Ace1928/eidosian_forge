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
def decompress(self, data):
    for d in reversed(self._decoders):
        data = d.decompress(data)
    return data