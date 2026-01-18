from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_previous(self):
    self.assert_selects(self.end.find_all_previous('b'), ['Three', 'Two', 'One'])
    self.assert_selects(self.end.find_all_previous(id=1), ['One'])