from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_httpWithoutHost(self):
    """
        An HTTP URL without a hostname, but with a path, should also round-trip
        cleanly.
        """
    without_host = URL.from_text('http:relative-path')
    self.assertEqual(without_host.host, '')
    self.assertEqual(without_host.path, ('relative-path',))
    self.assertEqual(without_host.uses_netloc, False)
    self.assertEqual(without_host.to_text(), 'http:relative-path')