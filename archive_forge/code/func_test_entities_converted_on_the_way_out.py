import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_entities_converted_on_the_way_out(self):
    text = '<p>&lt;&lt;sacr&eacute;&#32;bleu!&gt;&gt;</p>'
    expected = '<p>&lt;&lt;sacré bleu!&gt;&gt;</p>'.encode('utf-8')
    soup = self.soup(text)
    assert soup.p.encode('utf-8') == expected