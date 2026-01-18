import pytest
from bs4.element import (
from . import SoupTest
def test_cdata(self):
    soup = self.soup('')
    cdata = CData('foo')
    soup.insert(1, cdata)
    assert str(soup) == '<![CDATA[foo]]>'
    assert soup.find(string='foo') == 'foo'
    assert soup.contents[0] == 'foo'