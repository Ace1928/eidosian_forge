from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_by_utf8_attribute_value(self):
    peace = 'םולש'.encode('utf8')
    data = '<a title="םולש"></a>'.encode('utf8')
    soup = self.soup(data)
    assert [soup.a] == soup.find_all(title=peace)
    assert [soup.a] == soup.find_all(title=peace.decode('utf8'))
    assert [soup.a], soup.find_all(title=[peace, 'something else'])