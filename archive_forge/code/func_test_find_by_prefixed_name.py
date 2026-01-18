import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_find_by_prefixed_name(self):
    doc = '<?xml version="1.0" encoding="utf-8"?>\n<Document xmlns="http://example.com/ns0"\n    xmlns:ns1="http://example.com/ns1"\n    xmlns:ns2="http://example.com/ns2">\n    <ns1:tag>foo</ns1:tag>\n    <ns1:tag>bar</ns1:tag>\n    <ns2:tag key="value">baz</ns2:tag>\n</Document>\n'
    soup = self.soup(doc)
    assert 3 == len(soup.find_all('tag'))
    assert 2 == len(soup.find_all('ns1:tag'))
    assert 1 == len(soup.find_all('ns2:tag'))
    assert 1, len(soup.find_all('ns2:tag', key='value'))
    assert 3, len(soup.find_all(['ns1:tag', 'ns2:tag']))