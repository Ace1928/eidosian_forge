import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_copy_navigablestring_subclass_has_same_type(self):
    html = '<b><!--Foo--></b>'
    soup = self.soup(html)
    s1 = soup.string
    s2 = copy.copy(s1)
    assert s1 == s2
    assert isinstance(s2, Comment)