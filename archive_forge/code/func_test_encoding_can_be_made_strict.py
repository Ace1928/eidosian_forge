import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_encoding_can_be_made_strict(self):
    html = '<b>☃</b>'
    soup = self.soup(html)
    with pytest.raises(UnicodeEncodeError):
        soup.encode('ascii', errors='strict')