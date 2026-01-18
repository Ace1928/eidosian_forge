import pytest
import warnings
from bs4 import BeautifulSoup
from bs4.element import SoupStrainer
from . import (
def test_foster_parenting(self):
    markup = b'<table><td></tbody>A'
    soup = self.soup(markup)
    assert '<body>A<table><tbody><tr><td></td></tr></tbody></table></body>' == soup.body.decode()