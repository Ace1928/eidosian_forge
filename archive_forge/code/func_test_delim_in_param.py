from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_delim_in_param(self):
    """Per issue #6 and #8"""
    self.assertRaises(ValueError, URL, scheme='http', host='a/c')
    self.assertRaises(ValueError, URL, path=('?',))
    self.assertRaises(ValueError, URL, path=('#',))
    self.assertRaises(ValueError, URL, query=('&', 'test'))