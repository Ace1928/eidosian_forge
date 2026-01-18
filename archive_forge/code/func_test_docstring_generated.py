import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_docstring_generated(self):
    soup = self.soup('<root/>')
    assert soup.encode() == b'<?xml version="1.0" encoding="utf-8"?>\n<root/>'