from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_next_sibling_for_text_element(self):
    soup = self.soup('Foo<b>bar</b>baz')
    start = soup.find(string='Foo')
    assert start.next_sibling.name == 'b'
    assert start.next_sibling.next_sibling == 'baz'
    self.assert_selects(start.find_next_siblings('b'), ['bar'])
    assert start.find_next_sibling(string='baz') == 'baz'
    assert start.find_next_sibling(string='nonesuch') == None