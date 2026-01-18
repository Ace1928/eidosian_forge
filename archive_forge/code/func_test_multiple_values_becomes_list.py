import warnings
from bs4.element import (
from . import SoupTest
def test_multiple_values_becomes_list(self):
    soup = self.soup("<a class='foo bar'>")
    assert ['foo', 'bar'] == soup.a['class']