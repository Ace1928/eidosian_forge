import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_angle_brackets_in_attribute_values_are_escaped(self):
    self.assert_soup('<a b="<a>"></a>', '<a b="&lt;a&gt;"></a>')