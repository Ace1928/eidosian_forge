from your HTTP server.
import pprint
import re
import socket
import sys
import time
import traceback
import os
import json
import unittest  # pylint: disable=deprecated-module,preferred-module
import warnings
import functools
import http.client
import urllib.parse
from more_itertools.more import always_iterable
import jaraco.functools
def strip_netloc(url):
    """Return absolute-URI path from URL.

    Strip the scheme and host from the URL, returning the
    server-absolute portion.

    Useful for wrapping an absolute-URI for which only the
    path is expected (such as in calls to :py:meth:`WebCase.getPage`).

    .. testsetup::

       from cheroot.test.webtest import strip_netloc

    >>> strip_netloc('https://google.com/foo/bar?bing#baz')
    '/foo/bar?bing'

    >>> strip_netloc('//google.com/foo/bar?bing#baz')
    '/foo/bar?bing'

    >>> strip_netloc('/foo/bar?bing#baz')
    '/foo/bar?bing'
    """
    parsed = urllib.parse.urlparse(url)
    _scheme, _netloc, path, params, query, _fragment = parsed
    stripped = ('', '', path, params, query, '')
    return urllib.parse.urlunparse(stripped)