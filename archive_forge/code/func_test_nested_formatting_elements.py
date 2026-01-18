import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_nested_formatting_elements(self):
    self.assert_soup('<em><em></em></em>')