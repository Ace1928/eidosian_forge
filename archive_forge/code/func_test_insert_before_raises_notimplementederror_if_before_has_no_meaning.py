from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_insert_before_raises_notimplementederror_if_before_has_no_meaning(self):
    soup = self.soup('')
    tag = soup.new_tag('a')
    string = soup.new_string('')
    with pytest.raises(ValueError):
        string.insert_before(tag)
    with pytest.raises(NotImplementedError):
        soup.insert_before(tag)
    with pytest.raises(ValueError):
        tag.insert_before(tag)