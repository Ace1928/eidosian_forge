from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_with_missing_attribute(self):
    tree = self.soup('<a id="1">ID present.</a>\n                            <a>No ID present.</a>\n                            <a id="">ID is empty.</a>')
    self.assert_selects(tree.find_all('a', id=None), ['No ID present.'])