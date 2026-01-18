import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_mixed_case_doctype(self):
    for doctype_fragment in ('doctype', 'DocType'):
        doctype_str, soup = self._document_with_doctype('html', doctype_fragment)
        doctype = soup.contents[0]
        assert doctype.__class__ == Doctype
        assert doctype == 'html'
        assert soup.encode('utf8')[:len(doctype_str)] == b'<!DOCTYPE html>'
        assert soup.p.contents[0] == 'foo'