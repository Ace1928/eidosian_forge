import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_copy_deeply_nested_document(self):
    limit = sys.getrecursionlimit() + 1
    markup = '<span>' * limit
    soup = self.soup(markup)
    copied = copy.copy(soup)
    copied = copy.deepcopy(soup)