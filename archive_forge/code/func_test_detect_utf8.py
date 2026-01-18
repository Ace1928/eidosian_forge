import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_detect_utf8(self):
    utf8 = b'Sacr\xc3\xa9 bleu! \xe2\x98\x83'
    dammit = UnicodeDammit(utf8)
    assert dammit.original_encoding.lower() == 'utf-8'
    assert dammit.unicode_markup == 'Sacré bleu! ☃'