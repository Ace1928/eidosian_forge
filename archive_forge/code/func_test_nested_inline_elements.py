import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_nested_inline_elements(self):
    """Inline elements can be nested indefinitely."""
    b_tag = '<b>Inside a B tag</b>'
    self.assert_soup(b_tag)
    nested_b_tag = '<p>A <i>nested <b>tag</b></i></p>'
    self.assert_soup(nested_b_tag)
    double_nested_b_tag = '<p>A <a>doubly <i>nested <b>tag</b></i></a></p>'
    self.assert_soup(nested_b_tag)