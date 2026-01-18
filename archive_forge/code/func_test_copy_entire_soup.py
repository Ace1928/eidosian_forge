import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_copy_entire_soup(self):
    html = '<div><b>Foo<a></a></b><b>Bar</b></div>end'
    soup = self.soup(html)
    soup_copy = copy.copy(soup)
    assert soup == soup_copy