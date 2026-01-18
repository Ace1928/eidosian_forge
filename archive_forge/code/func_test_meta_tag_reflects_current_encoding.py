import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_meta_tag_reflects_current_encoding(self):
    meta_tag = '<meta content="text/html; charset=x-sjis" http-equiv="Content-type"/>'
    shift_jis_html = '<html><head>\n%s\n<meta http-equiv="Content-language" content="ja"/></head><body>Shift-JIS markup goes here.' % meta_tag
    soup = self.soup(shift_jis_html)
    parsed_meta = soup.find('meta', {'http-equiv': 'Content-type'})
    content = parsed_meta['content']
    assert 'text/html; charset=x-sjis' == content
    assert isinstance(content, ContentMetaAttributeValue)
    assert 'text/html; charset=utf8' == content.encode('utf8')