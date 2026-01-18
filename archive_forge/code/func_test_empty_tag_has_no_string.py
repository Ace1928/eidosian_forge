import warnings
from bs4.element import (
from . import SoupTest
def test_empty_tag_has_no_string(self):
    soup = self.soup('<b></b>')
    assert soup.b.string == None