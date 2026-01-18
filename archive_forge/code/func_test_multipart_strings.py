import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_multipart_strings(self):
    """Mostly to prevent a recurrence of a bug in the html5lib treebuilder."""
    soup = self.soup('<html><h2>\nfoo</h2><p></p></html>')
    assert 'p' == soup.h2.string.next_element.name
    assert 'p' == soup.p.name
    self.assertConnectedness(soup)