from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_encoded_userinfo(self):
    url = URL.from_text('http://user:pass@example.com')
    assert url.userinfo == 'user:pass'
    url = url.replace(userinfo='us%20her:pass')
    iri = url.to_iri()
    assert iri.to_text(with_password=True) == 'http://us her:pass@example.com'
    assert iri.to_text(with_password=False) == 'http://us her:@example.com'
    assert iri.to_uri().to_text(with_password=True) == 'http://us%20her:pass@example.com'