from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_with_regular_expression_attribute_value(self):
    tree = self.soup('<a id="a">One a.</a>\n                            <a id="aa">Two as.</a>\n                            <a id="ab">Mixed as and bs.</a>\n                            <a id="b">One b.</a>\n                            <a>No ID.</a>')
    self.assert_selects(tree.find_all(id=re.compile('^a+$')), ['One a.', 'Two as.'])