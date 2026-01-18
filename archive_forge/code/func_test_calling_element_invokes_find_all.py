from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
def test_calling_element_invokes_find_all(self):
    self.assert_selects(self.tree('a'), ['First tag.', 'Nested tag.'])