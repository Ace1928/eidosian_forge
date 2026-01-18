from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_append_child_thats_already_at_the_end(self):
    data = '<a><b></b></a>'
    soup = self.soup(data)
    soup.a.append(soup.b)
    assert data == soup.decode()