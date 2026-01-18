import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_encoding_substitutes_unrecognized_characters_by_default(self):
    html = '<b>☃</b>'
    soup = self.soup(html)
    assert soup.b.encode('ascii') == b'<b>&#9731;</b>'