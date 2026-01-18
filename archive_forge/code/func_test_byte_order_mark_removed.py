import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_byte_order_mark_removed(self):
    data = b'\xff\xfe<\x00a\x00>\x00\xe1\x00\xe9\x00<\x00/\x00a\x00>\x00'
    dammit = UnicodeDammit(data)
    assert '<a>áé</a>' == dammit.unicode_markup
    assert 'utf-16le' == dammit.original_encoding