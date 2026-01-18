import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_html5_style_meta_tag_reflects_current_encoding(self):
    meta_tag = '<meta id="encoding" charset="x-sjis" />'
    shift_jis_html = '<html><head>\n%s\n<meta http-equiv="Content-language" content="ja"/></head><body>Shift-JIS markup goes here.' % meta_tag
    soup = self.soup(shift_jis_html)
    parsed_meta = soup.find('meta', id='encoding')
    charset = parsed_meta['charset']
    assert 'x-sjis' == charset
    assert isinstance(charset, CharsetMetaAttributeValue)
    assert 'utf8' == charset.encode('utf8')