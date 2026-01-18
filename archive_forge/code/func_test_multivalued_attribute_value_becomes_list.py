import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_multivalued_attribute_value_becomes_list(self):
    markup = b'<a class="foo bar">'
    soup = self.soup(markup)
    assert ['foo', 'bar'] == soup.a['class']