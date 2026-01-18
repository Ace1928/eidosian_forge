from __future__ import annotations
import errno
import io
import os
import selectors
import socket
import socketserver
import sys
import typing as t
from datetime import datetime as dt
from datetime import timedelta
from datetime import timezone
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from urllib.parse import unquote
from urllib.parse import urlsplit
from ._internal import _log
from ._internal import _wsgi_encoding_dance
from .exceptions import InternalServerError
from .urls import uri_to_iri
def get_sockaddr(host: str, port: int, family: socket.AddressFamily) -> tuple[str, int] | str:
    """Return a fully qualified socket address that can be passed to
    :func:`socket.bind`."""
    if family == af_unix:
        return os.path.abspath(host.partition('://')[2])
    try:
        res = socket.getaddrinfo(host, port, family, socket.SOCK_STREAM, socket.IPPROTO_TCP)
    except socket.gaierror:
        return (host, port)
    return res[0][4]