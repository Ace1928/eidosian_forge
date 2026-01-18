from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_by_name_and_containing_string_when_string_is_buried(self):
    soup = self.soup('<a>foo</a><a><b><c>foo</c></b></a>')
    assert soup.find_all('a') == soup.find_all('a', string='foo')