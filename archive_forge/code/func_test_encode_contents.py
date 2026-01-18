import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_encode_contents(self):
    html = '<b>☃</b>'
    soup = self.soup(html)
    assert '☃'.encode('utf8') == soup.b.encode_contents(encoding='utf8')