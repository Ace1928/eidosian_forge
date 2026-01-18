from __future__ import unicode_literals
from typing import Dict, Union
from .. import DecodedURL, URL
from .._url import _percent_decode
from .common import HyperlinkTestCase
def test_equality_and_hashability(self):
    durl = DecodedURL.from_text(TOTAL_URL)
    durl2 = DecodedURL.from_text(TOTAL_URL)
    burl = DecodedURL.from_text(BASIC_URL)
    durl_uri = durl.to_uri()
    assert durl == durl
    assert durl == durl2
    assert durl != burl
    assert durl is not None
    assert durl != durl._url
    AnyURL = Union[URL, DecodedURL]
    durl_map = {}
    durl_map[durl] = durl
    durl_map[durl2] = durl2
    assert len(durl_map) == 1
    durl_map[burl] = burl
    assert len(durl_map) == 2
    durl_map[durl_uri] = durl_uri
    assert len(durl_map) == 3