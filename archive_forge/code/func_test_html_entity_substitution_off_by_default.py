import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_html_entity_substitution_off_by_default(self):
    markup = '<b>Sacré bleu!</b>'
    soup = self.soup(markup)
    encoded = soup.b.encode('utf-8')
    assert encoded == markup.encode('utf-8')