import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_can_parse_unicode_document(self):
    markup = '<?xml version="1.0" encoding="euc-jp"><root>Sacré bleu!</root>'
    soup = self.soup(markup)
    assert 'Sacré bleu!' == soup.root.string