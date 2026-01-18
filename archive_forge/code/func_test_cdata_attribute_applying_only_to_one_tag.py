import warnings
from bs4.element import (
from . import SoupTest
def test_cdata_attribute_applying_only_to_one_tag(self):
    data = '<a accept-charset="ISO-8859-1 UTF-8"></a>'
    soup = self.soup(data)
    assert 'ISO-8859-1 UTF-8' == soup.a['accept-charset']