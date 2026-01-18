import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_decode_contents(self):
    html = '<b>☃</b>'
    soup = self.soup(html)
    assert '☃' == soup.b.decode_contents()