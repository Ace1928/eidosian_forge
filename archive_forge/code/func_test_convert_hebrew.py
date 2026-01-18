import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
def test_convert_hebrew(self):
    hebrew = b'\xed\xe5\xec\xf9'
    dammit = UnicodeDammit(hebrew, ['iso-8859-8'])
    assert dammit.original_encoding.lower() == 'iso-8859-8'
    assert dammit.unicode_markup == 'םולש'