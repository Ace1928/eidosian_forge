import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_deprecated_renderContents(self):
    html = '<b>☃</b>'
    soup = self.soup(html)
    soup.renderContents()
    assert '☃'.encode('utf8') == soup.b.renderContents()