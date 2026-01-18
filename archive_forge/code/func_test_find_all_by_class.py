from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_find_all_by_class(self):
    tree = self.soup('\n                         <a class="1">Class 1.</a>\n                         <a class="2">Class 2.</a>\n                         <b class="1">Class 1.</b>\n                         <c class="3 4">Class 3 and 4.</c>\n                         ')
    self.assert_selects(tree.find_all('a', class_='1'), ['Class 1.'])
    self.assert_selects(tree.find_all('c', class_='3'), ['Class 3 and 4.'])
    self.assert_selects(tree.find_all('c', class_='4'), ['Class 3 and 4.'])
    self.assert_selects(tree.find_all('a', '1'), ['Class 1.'])
    self.assert_selects(tree.find_all(attrs='1'), ['Class 1.', 'Class 1.'])
    self.assert_selects(tree.find_all('c', '3'), ['Class 3 and 4.'])
    self.assert_selects(tree.find_all('c', '4'), ['Class 3 and 4.'])