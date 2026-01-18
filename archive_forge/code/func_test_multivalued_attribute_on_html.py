import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_multivalued_attribute_on_html(self):
    markup = '<html class="a b"></html>'
    soup = self.soup(markup)
    assert ['a', 'b'] == soup.html['class']