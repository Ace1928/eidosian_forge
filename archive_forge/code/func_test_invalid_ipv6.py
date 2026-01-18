from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_invalid_ipv6(self):
    invalid_ipv6_ips = ['2001::0234:C1ab::A0:aabc:003F', '2001::1::3F', ':', '::::', '::256.0.0.1']
    for ip in invalid_ipv6_ips:
        url_text = 'http://[' + ip + ']'
        self.assertRaises(socket.error, inet_pton, socket.AF_INET6, ip)
        self.assertRaises(URLParseError, URL.from_text, url_text)