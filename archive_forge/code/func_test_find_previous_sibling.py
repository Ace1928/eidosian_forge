from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_previous_sibling(self):
    assert self.end.find_previous_sibling('span')['id'] == '3'