import warnings
from bs4.element import (
from . import SoupTest
def test_multiple_values_separated_by_weird_whitespace(self):
    soup = self.soup("<a class='foo\tbar\nbaz'>")
    assert ['foo', 'bar', 'baz'] == soup.a['class']