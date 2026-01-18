from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_by_name_and_containing_string(self):
    soup = self.soup('<b>foo</b><b>bar</b><a>foo</a>')
    a = soup.a
    assert [a] == soup.find_all('a', string='foo')
    assert [] == soup.find_all('a', string='bar')