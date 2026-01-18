from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_by_attribute_name(self):
    tree = self.soup('\n                         <a id="first">Matching a.</a>\n                         <a id="second">\n                          Non-matching <b id="first">Matching b.</b>a.\n                         </a>')
    self.assert_selects(tree.find_all(id='first'), ['Matching a.', 'Matching b.'])