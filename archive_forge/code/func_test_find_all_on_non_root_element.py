from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_on_non_root_element(self):
    self.assert_selects(self.tree.c.find_all('a'), ['Nested tag.'])