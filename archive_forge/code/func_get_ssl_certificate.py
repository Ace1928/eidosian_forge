import calendar
import collections.abc
import copy
import datetime
import email.utils
from functools import lru_cache
from http.client import responses
import http.cookies
import re
from ssl import SSLError
import time
import unicodedata
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from tornado.escape import native_str, parse_qs_bytes, utf8
from tornado.log import gen_log
from tornado.util import ObjectDict, unicode_type
import typing
from typing import (
def get_ssl_certificate(self, binary_form: bool=False) -> Union[None, Dict, bytes]:
    """Returns the client's SSL certificate, if any.

        To use client certificates, the HTTPServer's
        `ssl.SSLContext.verify_mode` field must be set, e.g.::

            ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_ctx.load_cert_chain("foo.crt", "foo.key")
            ssl_ctx.load_verify_locations("cacerts.pem")
            ssl_ctx.verify_mode = ssl.CERT_REQUIRED
            server = HTTPServer(app, ssl_options=ssl_ctx)

        By default, the return value is a dictionary (or None, if no
        client certificate is present).  If ``binary_form`` is true, a
        DER-encoded form of the certificate is returned instead.  See
        SSLSocket.getpeercert() in the standard library for more
        details.
        http://docs.python.org/library/ssl.html#sslsocket-objects
        """
    try:
        if self.connection is None:
            return None
        return self.connection.stream.socket.getpeercert(binary_form=binary_form)
    except SSLError:
        return None