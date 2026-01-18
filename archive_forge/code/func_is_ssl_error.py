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
def is_ssl_error(error: Exception | None=None) -> bool:
    """Checks if the given error (or the current one) is an SSL error."""
    if error is None:
        error = t.cast(Exception, sys.exc_info()[1])
    return isinstance(error, ssl.SSLError)