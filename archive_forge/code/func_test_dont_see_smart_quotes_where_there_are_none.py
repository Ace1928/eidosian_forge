import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_dont_see_smart_quotes_where_there_are_none(self):
    utf_8 = b'\xe3\x82\xb1\xe3\x83\xbc\xe3\x82\xbf\xe3\x82\xa4 Watch'
    dammit = UnicodeDammit(utf_8)
    assert dammit.original_encoding.lower() == 'utf-8'
    assert dammit.unicode_markup.encode('utf-8') == utf_8