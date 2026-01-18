import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_basic_namespaces(self):
    """Parsers don't need to *understand* namespaces, but at the
        very least they should not choke on namespaces or lose
        data."""
    markup = b'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:mathml="http://www.w3.org/1998/Math/MathML" xmlns:svg="http://www.w3.org/2000/svg"><head></head><body><mathml:msqrt>4</mathml:msqrt><b svg:fill="red"></b></body></html>'
    soup = self.soup(markup)
    assert markup == soup.encode()
    html = soup.html
    assert 'http://www.w3.org/1999/xhtml' == soup.html['xmlns']
    assert 'http://www.w3.org/1998/Math/MathML' == soup.html['xmlns:mathml']
    assert 'http://www.w3.org/2000/svg' == soup.html['xmlns:svg']