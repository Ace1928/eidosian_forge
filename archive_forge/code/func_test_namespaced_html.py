import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_namespaced_html(self):
    markup = b'<ns1:foo>content</ns1:foo><ns1:foo/><ns2:foo/>'
    with warnings.catch_warnings(record=True) as w:
        soup = self.soup(markup)
    assert 2 == len(soup.find_all('ns1:foo'))
    assert [] == w