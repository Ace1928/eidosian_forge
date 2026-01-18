from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_text_nodes(self):
    """You can search the tree for text nodes."""
    soup = self.soup('<html>Foo<b>bar</b>»</html>')
    assert soup.find_all(string='bar') == ['bar']
    assert soup.find_all(string=['Foo', 'bar']) == ['Foo', 'bar']
    assert soup.find_all(string=re.compile('.*')) == ['Foo', 'bar', '»']
    assert soup.find_all(string=True) == ['Foo', 'bar', '»']