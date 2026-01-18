import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_entities_in_strings_converted_during_parsing(self):
    text = '<p>&lt;&lt;sacr&eacute;&#32;bleu!&gt;&gt;</p>'
    expected = '<p>&lt;&lt;sacré bleu!&gt;&gt;</p>'
    self.assert_soup(text, expected)