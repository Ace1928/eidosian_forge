from __future__ import unicode_literals
from typing import Dict, Union
from .. import DecodedURL, URL
from .._url import _percent_decode
from .common import HyperlinkTestCase
def test_click_decoded_url(self):
    durl = DecodedURL.from_text(TOTAL_URL)
    durl_dest = DecodedURL.from_text('/tëst')
    clicked = durl.click(durl_dest)
    assert clicked.host == durl.host
    assert clicked.path == durl_dest.path
    assert clicked.path == ('tëst',)