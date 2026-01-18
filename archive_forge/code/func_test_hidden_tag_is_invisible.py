import warnings
from bs4.element import (
from . import SoupTest
def test_hidden_tag_is_invisible(self):
    soup = self.soup('<div id="1"><span id="2">a string</span></div>')
    soup.span.hidden = True
    assert '<div id="1">a string</div>' == str(soup.div)