import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_formatter_custom(self):
    markup = '<b>&lt;foo&gt;</b><b>bar</b><br/>'
    soup = self.soup(markup)
    decoded = soup.decode(formatter=lambda x: x.upper())
    assert decoded == self.document_for('<b><FOO></b><b>BAR</b><br/>')