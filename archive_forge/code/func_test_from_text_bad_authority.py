from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_from_text_bad_authority(self):
    self.assertRaises(URLParseError, URL.from_text, 'http://[::1/')
    self.assertRaises(URLParseError, URL.from_text, 'http://::1]/')
    self.assertRaises(URLParseError, URL.from_text, 'http://[[::1]/')
    self.assertRaises(URLParseError, URL.from_text, 'http://[::1]]/')
    self.assertRaises(URLParseError, URL.from_text, 'http://127.0.0.1:')
    self.assertRaises(URLParseError, URL.from_text, 'http://127.0.0.1:hi')
    self.assertRaises(URLParseError, URL.from_text, 'http://127.0.0.1::80')