from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_from_text_type(self):
    assert URL.from_text('#ok').fragment == 'ok'
    self.assertRaises(TypeError, URL.from_text, b'bytes://x.y.z')
    self.assertRaises(TypeError, URL.from_text, object())