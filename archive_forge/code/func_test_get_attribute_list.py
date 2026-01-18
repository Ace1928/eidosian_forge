import warnings
from bs4.element import (
from . import SoupTest
def test_get_attribute_list(self):
    soup = self.soup("<a id='abc def'>")
    assert ['abc def'] == soup.a.get_attribute_list('id')