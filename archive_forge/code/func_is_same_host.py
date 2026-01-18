from __future__ import absolute_import
import errno
import logging
import re
import socket
import sys
import warnings
from socket import error as SocketError
from socket import timeout as SocketTimeout
from .connection import (
from .exceptions import (
from .packages import six
from .packages.six.moves import queue
from .request import RequestMethods
from .response import HTTPResponse
from .util.connection import is_connection_dropped
from .util.proxy import connection_requires_http_tunnel
from .util.queue import LifoQueue
from .util.request import set_file_position
from .util.response import assert_header_parsing
from .util.retry import Retry
from .util.ssl_match_hostname import CertificateError
from .util.timeout import Timeout
from .util.url import Url, _encode_target
from .util.url import _normalize_host as normalize_host
from .util.url import get_host, parse_url
def is_same_host(self, url):
    """
        Check if the given ``url`` is a member of the same host as this
        connection pool.
        """
    if url.startswith('/'):
        return True
    scheme, host, port = get_host(url)
    if host is not None:
        host = _normalize_host(host, scheme=scheme)
    if self.port and (not port):
        port = port_by_scheme.get(scheme)
    elif not self.port and port == port_by_scheme.get(scheme):
        port = None
    return (scheme, host, port) == (self.scheme, self.host, self.port)