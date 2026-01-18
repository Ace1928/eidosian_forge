from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_next_sibling_may_not_exist(self):
    assert self.tree.html.next_sibling == None
    nested_span = self.tree.find(id='1.1')
    assert nested_span.next_sibling == None
    last_span = self.tree.find(id='4')
    assert last_span.next_sibling == None