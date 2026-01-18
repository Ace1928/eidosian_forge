from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_by_attribute_dict(self):
    tree = self.soup('\n                         <a name="name1" class="class1">Name match.</a>\n                         <a name="name2" class="class2">Class match.</a>\n                         <a name="name3" class="class3">Non-match.</a>\n                         <name1>A tag called \'name1\'.</name1>\n                         ')
    self.assert_selects(tree.find_all(name='name1'), ["A tag called 'name1'."])
    self.assert_selects(tree.find_all(attrs={'name': 'name1'}), ['Name match.'])
    self.assert_selects(tree.find_all(attrs={'class': 'class2'}), ['Class match.'])