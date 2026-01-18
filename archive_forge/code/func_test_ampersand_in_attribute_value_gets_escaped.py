import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_ampersand_in_attribute_value_gets_escaped(self):
    self.assert_soup('<this is="really messed up & stuff"></this>', '<this is="really messed up &amp; stuff"></this>')
    self.assert_soup('<a href="http://example.org?a=1&b=2;3">foo</a>', '<a href="http://example.org?a=1&amp;b=2;3">foo</a>')