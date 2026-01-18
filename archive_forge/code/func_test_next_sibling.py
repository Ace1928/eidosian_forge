from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_next_sibling(self):
    assert self.start.next_sibling['id'] == '2'
    assert self.start.next_sibling.next_sibling['id'] == '3'
    assert self.start.next_element['id'] == '1.1'