import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def assertDoctypeHandled(self, doctype_fragment):
    """Assert that a given doctype string is handled correctly."""
    doctype_str, soup = self._document_with_doctype(doctype_fragment)
    doctype = soup.contents[0]
    assert doctype.__class__ == Doctype
    assert doctype == doctype_fragment
    assert soup.encode('utf8')[:len(doctype_str)] == doctype_str
    assert soup.p.contents[0] == 'foo'