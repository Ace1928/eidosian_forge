from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_with_list_attribute_values(self):
    tree = self.soup('<a id="1">1</a>\n                            <a id="2">2</a>\n                            <a id="3">3</a>\n                            <a>No ID.</a>')
    self.assert_selects(tree.find_all(id=['1', '3', '4']), ['1', '3'])