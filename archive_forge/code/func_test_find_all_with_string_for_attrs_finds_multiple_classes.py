from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_with_string_for_attrs_finds_multiple_classes(self):
    soup = self.soup('<a class="foo bar"></a><a class="foo"></a>')
    a, a2 = soup.find_all('a')
    assert [a, a2], soup.find_all('a', 'foo')
    assert [a], soup.find_all('a', 'bar')
    assert [a] == soup.find_all('a', class_='foo bar')
    assert [a] == soup.find_all('a', 'foo bar')
    assert [] == soup.find_all('a', 'bar foo')