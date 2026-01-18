import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def test_entities_in_text_converted_to_unicode(self):
    expect = '<p>piñata</p>'
    self.assert_soup('<p>pi&#241;ata</p>', expect)
    self.assert_soup('<p>pi&#xf1;ata</p>', expect)
    self.assert_soup('<p>pi&#Xf1;ata</p>', expect)
    self.assert_soup('<p>pi&ntilde;ata</p>', expect)