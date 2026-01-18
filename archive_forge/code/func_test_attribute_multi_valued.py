import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
@pytest.mark.parametrize('multi_valued_attributes', [dict(a=['class']), {'*': ['class']}])
def test_attribute_multi_valued(self, multi_valued_attributes):
    markup = '<a class="a b c">'
    soup = self.soup(markup, multi_valued_attributes=multi_valued_attributes)
    assert soup.a['class'] == ['a', 'b', 'c']