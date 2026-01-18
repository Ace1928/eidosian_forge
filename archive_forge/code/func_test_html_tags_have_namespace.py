import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_html_tags_have_namespace(self):
    markup = '<a>'
    soup = self.soup(markup)
    assert 'http://www.w3.org/1999/xhtml' == soup.a.namespace