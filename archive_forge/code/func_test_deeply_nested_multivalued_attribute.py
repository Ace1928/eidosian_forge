import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_deeply_nested_multivalued_attribute(self):
    markup = '<table><div><div class="css"></div></div></table>'
    soup = self.soup(markup)
    assert ['css'] == soup.div.div['class']