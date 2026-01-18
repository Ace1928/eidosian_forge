from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_next(self):
    self.assert_selects(self.start.find_all_next('b'), ['Two', 'Three'])
    self.start.find_all_next(id=3)
    self.assert_selects(self.start.find_all_next(id=3), ['Three'])