import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_large_xml_document(self):
    """A large XML document should come out the same as it went in."""
    markup = b'<?xml version="1.0" encoding="utf-8"?>\n<root>' + b'0' * 2 ** 12 + b'</root>'
    soup = self.soup(markup)
    assert soup.encode('utf-8') == markup