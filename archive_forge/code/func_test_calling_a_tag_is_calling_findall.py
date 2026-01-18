from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_calling_a_tag_is_calling_findall(self):
    soup = self.soup("<a>1</a><b>2<a id='foo'>3</a></b>")
    self.assert_selects(soup('a', limit=1), ['1'])
    self.assert_selects(soup.b(id='foo'), ['3'])