import copy
import pickle
import pytest
import sys
from bs4 import BeautifulSoup
from bs4.element import (
from . import (
def test_formatter_is_run_on_attribute_values(self):
    markup = '<a href="http://a.com?a=b&c=é">e</a>'
    soup = self.soup(markup)
    a = soup.a
    expect_minimal = '<a href="http://a.com?a=b&amp;c=é">e</a>'
    assert expect_minimal == a.decode()
    assert expect_minimal == a.decode(formatter='minimal')
    expect_html = '<a href="http://a.com?a=b&amp;c=&eacute;">e</a>'
    assert expect_html == a.decode(formatter='html')
    assert markup == a.decode(formatter=None)
    expect_upper = '<a href="HTTP://A.COM?A=B&C=É">E</a>'
    assert expect_upper == a.decode(formatter=lambda x: x.upper())