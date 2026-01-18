from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_next_generator(self):
    start = self.tree.find(string='Two')
    successors = [node for node in start.next_elements]
    tag, contents = successors
    assert tag['id'] == '3'
    assert contents == 'Three'