from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_empty_paths_eq(self):
    u1 = URL.from_text('http://example.com/')
    u2 = URL.from_text('http://example.com')
    assert u1 == u2
    u1 = URL.from_text('http://example.com')
    u2 = URL.from_text('http://example.com')
    assert u1 == u2
    u1 = URL.from_text('http://example.com')
    u2 = URL.from_text('http://example.com/')
    assert u1 == u2
    u1 = URL.from_text('http://example.com/')
    u2 = URL.from_text('http://example.com/')
    assert u1 == u2