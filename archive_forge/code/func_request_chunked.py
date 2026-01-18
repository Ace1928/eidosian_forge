from __future__ import absolute_import
import datetime
import logging
import os
import re
import socket
import warnings
from socket import error as SocketError
from socket import timeout as SocketTimeout
from .packages import six
from .packages.six.moves.http_client import HTTPConnection as _HTTPConnection
from .packages.six.moves.http_client import HTTPException  # noqa: F401
from .util.proxy import create_proxy_ssl_context
from ._collections import HTTPHeaderDict  # noqa (historical, removed in v2)
from ._version import __version__
from .exceptions import (
from .util import SKIP_HEADER, SKIPPABLE_HEADERS, connection
from .util.ssl_ import (
from .util.ssl_match_hostname import CertificateError, match_hostname
def request_chunked(self, method, url, body=None, headers=None):
    """
        Alternative to the common request method, which sends the
        body with chunked encoding and not as one block
        """
    headers = headers or {}
    header_keys = set([six.ensure_str(k.lower()) for k in headers])
    skip_accept_encoding = 'accept-encoding' in header_keys
    skip_host = 'host' in header_keys
    self.putrequest(method, url, skip_accept_encoding=skip_accept_encoding, skip_host=skip_host)
    if 'user-agent' not in header_keys:
        self.putheader('User-Agent', _get_default_user_agent())
    for header, value in headers.items():
        self.putheader(header, value)
    if 'transfer-encoding' not in header_keys:
        self.putheader('Transfer-Encoding', 'chunked')
    self.endheaders()
    if body is not None:
        stringish_types = six.string_types + (bytes,)
        if isinstance(body, stringish_types):
            body = (body,)
        for chunk in body:
            if not chunk:
                continue
            if not isinstance(chunk, bytes):
                chunk = chunk.encode('utf8')
            len_str = hex(len(chunk))[2:]
            to_send = bytearray(len_str.encode())
            to_send += b'\r\n'
            to_send += chunk
            to_send += b'\r\n'
            self.send(to_send)
    self.send(b'0\r\n\r\n')