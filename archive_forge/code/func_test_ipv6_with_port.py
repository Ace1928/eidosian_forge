from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_ipv6_with_port(self):
    t = 'https://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:80/'
    url = URL.from_text(t)
    assert url.host == '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    assert url.port == 80
    assert SCHEME_PORT_MAP[url.scheme] != url.port