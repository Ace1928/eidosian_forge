from __future__ import annotations
import collections
import os
import warnings
import zlib
from dataclasses import dataclass
from functools import wraps
from http.cookiejar import CookieJar
from typing import TYPE_CHECKING, Iterable, Optional
from urllib.parse import urldefrag, urljoin, urlunparse as _urlunparse
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, protocol, task
from twisted.internet.abstract import isIPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.endpoints import HostnameEndpoint, wrapClientTLS
from twisted.internet.interfaces import IOpenSSLContextFactory, IProtocol
from twisted.logger import Logger
from twisted.python.compat import nativeString, networkString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import (
from twisted.python.failure import Failure
from twisted.web import error, http
from twisted.web._newclient import _ensureValidMethod, _ensureValidURI
from twisted.web.http_headers import Headers
from twisted.web.iweb import (
from twisted.web._newclient import (
from twisted.web.error import SchemeNotSupported
@property
def originForm(self):
    """
        The absolute I{URI} path including I{URI} parameters, query string and
        fragment identifier.

        @see: U{https://tools.ietf.org/html/draft-ietf-httpbis-p1-messaging-21#section-5.3}

        @return: The absolute path in original form.
        @rtype: L{bytes}
        """
    path = urlunparse((b'', b'', self.path, self.params, self.query, b''))
    if path == b'':
        path = b'/'
    return path