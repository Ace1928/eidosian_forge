import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_correctly_nested_tables(self):
    """One table can go inside another one."""
    markup = '<table id="1"><tr><td>Here\'s another table:<table id="2"><tr><td>foo</td></tr></table></td>'
    self.assert_soup(markup, '<table id="1"><tr><td>Here\'s another table:<table id="2"><tr><td>foo</td></tr></table></td></tr></table>')
    self.assert_soup('<table><thead><tr><td>Foo</td></tr></thead><tbody><tr><td>Bar</td></tr></tbody><tfoot><tr><td>Baz</td></tr></tfoot></table>')