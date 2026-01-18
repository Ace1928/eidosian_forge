from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_insert_string(self):
    soup = self.soup('<a></a>')
    soup.a.insert(0, 'bar')
    soup.a.insert(0, 'foo')
    assert ['foo', 'bar'] == soup.a.contents
    assert soup.a.contents[0].next_element == 'bar'