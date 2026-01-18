import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_apos_entity(self):
    self.assert_soup('<p>Bob&apos;s Bar</p>', "<p>Bob's Bar</p>")