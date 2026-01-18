import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_detect_html5_style_meta_tag(self):
    for data in (b'<html><meta charset="euc-jp" /></html>', b"<html><meta charset='euc-jp' /></html>", b'<html><meta charset=euc-jp /></html>', b'<html><meta charset=euc-jp/></html>'):
        dammit = UnicodeDammit(data, is_html=True)
        assert 'euc-jp' == dammit.original_encoding